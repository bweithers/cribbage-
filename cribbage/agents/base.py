"""The agent interface.

An agent only ever sees an :class:`~cribbage.engine.InfoState` -- never the
engine's :class:`~cribbage.engine.CribbageState`.  That is enforced by the arena
rather than by convention, so an agent physically cannot peek at the opponent's
hand.

Two decisions, dispatched by :meth:`Agent.act`:

* ``discard`` returns a two-card tuple to lay away.
* ``play`` returns one card to lay down.
"""

from __future__ import annotations

from typing import Sequence

from ..engine import InfoState, Phase

__all__ = ["Agent"]


class Agent:
    """Base class for cribbage agents."""

    name: str = "agent"

    def act(self, info: InfoState):
        """Route a decision to :meth:`discard` or :meth:`play`."""
        if info.phase is Phase.DISCARD:
            return self.discard(info)
        return self.play(info)

    def discard(self, info: InfoState) -> Sequence[int]:
        raise NotImplementedError

    def play(self, info: InfoState) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        """Hook for agents carrying per-game state.  Default does nothing."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"
