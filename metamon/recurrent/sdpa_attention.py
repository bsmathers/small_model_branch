"""PyTorch SDPA attention with AMAGO-compatible recurrent KV caches.

Unlike AMAGO's optional ``flash_attn`` backend, this implementation uses
``torch.nn.functional.scaled_dot_product_attention`` and therefore dispatches
through PyTorch's native MPS backend on Apple silicon.  The module deliberately
keeps AMAGO's parameter-free attention interface so released checkpoints remain
strictly load-compatible.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import gin

from amago.nets.transformer import SelfAttention


@gin.configurable
class SDPAAttention(SelfAttention):
    """Causal SDPA with FlashAttention-compatible sliding-window semantics."""

    def __init__(
        self,
        causal: bool = True,
        dropout: float = 0.0,
        window_size: tuple[int, int] = (32, 0),
    ):
        super().__init__(causal=causal, dropout=dropout)
        self.window_size = window_size

    def _window_mask(self, length: int, device: torch.device) -> torch.Tensor | None:
        left, right = self.window_size
        if left < 0 and right < 0:
            return None
        query = torch.arange(length, device=device)[:, None]
        key = torch.arange(length, device=device)[None, :]
        mask = torch.ones((length, length), dtype=torch.bool, device=device)
        if left >= 0:
            mask &= key >= query - left
        if right >= 0:
            mask &= key <= query + right
        if self.causal:
            mask &= key <= query
        return mask

    def _forward_without_cache(self, qkv: torch.Tensor) -> torch.Tensor:
        q, k, v = (tensor.transpose(1, 2) for tensor in qkv.unbind(dim=2))
        mask = self._window_mask(q.shape[-2], q.device)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal and mask is None,
        )
        return out.transpose(1, 2)

    def _inference_with_cache(
        self,
        qkv: torch.Tensor,
        key_cache: torch.Tensor,
        val_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v = qkv.unbind(dim=2)
        if q.shape[1] != 1:
            raise ValueError("cached SDPA inference requires one query token")
        batch = torch.arange(q.shape[0], device=q.device)
        key_cache[batch, cache_seqlens] = k[:, 0]
        val_cache[batch, cache_seqlens] = v[:, 0]

        end = cache_seqlens + 1
        # Attend over the compact fixed-capacity cache and mask unwritten
        # slots. Avoiding Tensor.item() prevents an MPS synchronization and a
        # torch.compile graph break on every decoder layer.
        max_len = key_cache.shape[1]
        keys = torch.nan_to_num(key_cache).transpose(1, 2)
        values = torch.nan_to_num(val_cache).transpose(1, 2)
        query = q.transpose(1, 2)

        positions = torch.arange(max_len, device=q.device)[None, :]
        valid = positions < end[:, None]
        left, _ = self.window_size
        if left >= 0:
            valid &= positions >= (end - left - 1)[:, None]
        # Every cached key precedes or equals the one-token query, so passing
        # is_causal=True here would incorrectly keep only key zero for Q=1.
        out = F.scaled_dot_product_attention(
            query,
            keys,
            values,
            attn_mask=valid[:, None, None, :],
            dropout_p=0.0,
            is_causal=False,
        )
        return out.transpose(1, 2)

    def forward(self, qkv, key_cache=None, val_cache=None, cache_seqlens=None):
        if key_cache is None or val_cache is None or cache_seqlens is None:
            return self._forward_without_cache(qkv)
        if self.training:
            raise ValueError("KV-cache inference requires eval mode")
        return self._inference_with_cache(qkv, key_cache, val_cache, cache_seqlens)
