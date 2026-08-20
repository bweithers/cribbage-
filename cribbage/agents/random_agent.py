"""Uniformly random play.  The floor that every other agent must clear."""

from __future__ import annotations

import random
from typing import Optional, Sequence

from ..engine import InfoState
from .base import Agent

__all__ = ["RandomAgent"]


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: Optional[int] = None, rng: Optional[random.Random] = None):
        self.rng = rng if rng is not None else random.Random(seed)

    def discard(self, info: InfoState) -> Sequence[int]:
        return self.rng.choice(info.legal)

    def play(self, info: InfoState) -> int:
        return self.rng.choice(info.legal)
