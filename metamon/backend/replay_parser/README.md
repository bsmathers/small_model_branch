# Replay Parser

The replay parser converts from the spectator POV of raw Pokémon Showdown replays to the first-peron POV of RL agents. In Showdown terms, this is a from-scratch python implementation of the sim protocol that simulates requests and predicts teams.

The process is summarized by the following (simplified) example.

<p align="center">
  <img src="../../../media/replay_reconstruction_example.png">
</p>

For a real example, view [this replay](https://replay.Pokémonshowdown.com/gen4nu-776588848). The parser begins with the raw replay:

<p align="center">
  <img src="../../../media/raw_replay_example.png" width="400">
</p>

And infers missing information to produce training data like:

<p align="center">
  <img src="../../../media/reconstructed_replay_example.png" width="400">
</p>

More information and discussion in Appendix D of the paper.