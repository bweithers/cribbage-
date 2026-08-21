"""Controls that isolate one part of a policy by wrecking another.

Comparing two agents that differ in several ways tells you little.  These differ
from :class:`~cribbage.agents.heuristic.HeuristicAgent` in exactly one respect
each, so a match against it measures that one thing.
"""

from __future__ import annotations

import random
from typing import Optional

from ..engine import InfoState
from .heuristic import HeuristicAgent

__all__ = ["RandomPlayAgent", "RandomDiscardAgent"]


class RandomPlayAgent(HeuristicAgent):
    """The heuristic discard, but the cards are laid down at random.

    Measures what the play policy is worth in total: everything else about the
    agent is unchanged, so the whole difference is pegging.
    """

    name = "randomplay"

    def __init__(self, seed: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.rng = random.Random(seed)

    def play(self, info: InfoState) -> int:
        return self.rng.choice(info.legal)


class RandomDiscardAgent(HeuristicAgent):
    """The heuristic play, but the lay-away is chosen at random.

    The mirror image, for scale: it says how the two halves of a cribbage policy
    compare in importance.
    """

    name = "randomdiscard"

    def __init__(self, seed: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.rng = random.Random(seed)

    def discard(self, info: InfoState):
        return self.rng.choice(info.legal)
