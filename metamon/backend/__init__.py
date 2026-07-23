import os
import re

BACKEND_PATH = os.path.dirname(__file__)


def format_to_gen(format: str) -> int:
    pattern = r"gen(\d+)"
    match = re.search(pattern, format.lower())
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract generation from format: {format}")


from . import replay_parser
