"""Exact solution of a pegging phase, given both hands.

The play is small.  Four cards each, at most eight plies, at most four choices at
a node -- so once the opponent's holding is known there is no need to guess at
all: the subgame can be solved outright by minimax on the net pegging
differential.

The opponent's holding is of course *not* known.  The way to use this is to
sample holdings consistent with what has been seen, solve each sampled world
exactly, and average the results per candidate card.  That is Perfect
Information Monte Carlo, and it is what
:class:`~cribbage.agents.pegging.PimcPeggingAgent` does.

**This is a second implementation of the play rules**, which is exactly the sort
of duplication the rest of this project avoids.  It is here because driving the
real engine through a few thousand search nodes is far too slow, and it is kept
honest the same way the scorer is: ``tests/test_pegsolve.py`` plays the solver's
own chosen line through the real engine and asserts the points that actually
come out match the value the solver predicted.  A divergence in the go rules,
the reset rules or the last-card rule would fail that test.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .cards import CARD_VALUES
from .engine import MAX_COUNT
from .scoring import score_play

__all__ = ["solve", "best_cards"]

ME = 0
THEM = 1


def _solve(
    hands: tuple[tuple[int, ...], tuple[int, ...]],
    count: int,
    seq: tuple[int, ...],
    turn: int,
    last: Optional[int],
    memo: dict,
) -> int:
    """Net pegging points to :data:`ME` from here, both sides playing best."""
    key = (hands, count, seq, turn, last)
    cached = memo.get(key)
    if cached is not None:
        return cached

    if not hands[0] and not hands[1]:
        # The play is over; a non-empty sequence means the last card did not
        # make 31, so whoever laid it takes a point for it.
        result = 0 if not seq or last is None else (1 if last == ME else -1)
        memo[key] = result
        return result

    room = MAX_COUNT - count
    playable = [card for card in hands[turn] if CARD_VALUES[card] <= room]

    if playable:
        best: Optional[int] = None
        for card in playable:
            extended = seq + (card,)
            scored = score_play(extended)
            gain = scored if turn == ME else -scored

            remaining = list(hands)
            remaining[turn] = tuple(c for c in hands[turn] if c != card)
            after = (remaining[0], remaining[1])
            total = count + CARD_VALUES[card]

            if total == MAX_COUNT:
                # Thirty-one resets at once, which is what stops the go branch
                # from paying for it a second time.
                value = gain + _solve(after, 0, (), 1 - turn, turn, memo)
            else:
                value = gain + _solve(after, total, extended, 1 - turn, turn, memo)

            if best is None or (value > best if turn == ME else value < best):
                best = value
        memo[key] = best
        return best

    other = 1 - turn
    if any(CARD_VALUES[card] <= room for card in hands[other]):
        # A go is not a turn change: the other player simply carries on.
        result = _solve(hands, count, seq, other, last, memo)
        memo[key] = result
        return result

    # Neither can play. The count cannot be 31 here, since that resets above.
    assert last is not None, "a go with no card played"
    go = 1 if last == ME else -1
    result = go + _solve(hands, 0, (), 1 - last, last, memo)
    memo[key] = result
    return result


def solve(
    my_hand: Sequence[int],
    their_hand: Sequence[int],
    count: int = 0,
    seq: Sequence[int] = (),
    my_turn: bool = True,
    last: Optional[int] = None,
    memo: Optional[dict] = None,
) -> int:
    """Net pegging points to me from this position, played out perfectly.

    ``last`` is who laid the most recent card overall (0 = me, 1 = them), which
    the go and last-card rules need; ``None`` if nothing has been played.
    """
    return _solve(
        (tuple(sorted(my_hand)), tuple(sorted(their_hand))),
        count, tuple(seq), ME if my_turn else THEM, last,
        {} if memo is None else memo,
    )


def best_cards(
    my_hand: Sequence[int],
    their_hand: Sequence[int],
    count: int = 0,
    seq: Sequence[int] = (),
    last: Optional[int] = None,
) -> dict[int, int]:
    """Value of each legal card for me, assuming best play thereafter."""
    memo: dict = {}
    room = MAX_COUNT - count
    values = {}
    for card in my_hand:
        if CARD_VALUES[card] > room:
            continue
        extended = tuple(seq) + (card,)
        gain = score_play(extended)
        rest = tuple(c for c in my_hand if c != card)
        total = count + CARD_VALUES[card]
        if total == MAX_COUNT:
            values[card] = gain + _solve(
                (tuple(sorted(rest)), tuple(sorted(their_hand))),
                0, (), THEM, ME, memo,
            )
        else:
            values[card] = gain + _solve(
                (tuple(sorted(rest)), tuple(sorted(their_hand))),
                total, extended, THEM, ME, memo,
            )
    return values
