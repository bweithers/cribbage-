"""Scoring: the show (hand and crib) and the play (pegging).

Everything a future search or net does is bottlenecked on these two functions,
so hand scoring is split into a part that can be precomputed and a part that
cannot:

* **Rank-only** -- fifteens, pairs and runs depend solely on the multiset of
  ranks.  There are only 6175 such multisets for five cards (``C(17,5) = 6188``
  five-card multisets over 13 ranks, less the 13 impossible five-of-a-kinds), so
  they are all enumerated at import into :data:`RANK_TABLE`.  Building it costs
  a few milliseconds; afterwards that whole side of hand scoring is one dict
  lookup.
* **Suit-dependent** -- flush and nobs need to know the suits and which card was
  the starter, so they are computed inline per call.  They are two cheap
  branches.

The play cannot be tabulated the same way because it depends on the *order*
cards were laid down, so :func:`score_play` scans the tail of the sequence
directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Sequence

from .cards import CARD_VALUES, JACK, RANK_VALUES

__all__ = [
    "HandScore",
    "PlayScore",
    "RANK_TABLE",
    "score_hand",
    "score_hand_detail",
    "score_play",
    "score_play_detail",
]


@dataclass(frozen=True)
class HandScore:
    """Breakdown of a four-card hand plus starter.  Used by tests and transcripts."""

    fifteens: int = 0
    pairs: int = 0
    runs: int = 0
    flush: int = 0
    nobs: int = 0

    @property
    def total(self) -> int:
        return self.fifteens + self.pairs + self.runs + self.flush + self.nobs

    def describe(self) -> str:
        """Human-readable summary like ``"fifteens 8, pairs 6, runs 3 = 17"``."""
        parts = [
            (name, getattr(self, name))
            for name in ("fifteens", "pairs", "runs", "flush", "nobs")
        ]
        shown = ", ".join(f"{name} {pts}" for name, pts in parts if pts)
        return f"{shown or 'nothing'} = {self.total}"


@dataclass(frozen=True)
class PlayScore:
    """Breakdown of the points scored by laying one card down during the play.

    The go / last-card point is *not* here: it is awarded by the engine when a
    sub-round ends, not by playing a card.
    """

    fifteen: int = 0
    thirty_one: int = 0
    pairs: int = 0
    run: int = 0

    @property
    def total(self) -> int:
        return self.fifteen + self.thirty_one + self.pairs + self.run


def _score_rank_multiset(ranks: Sequence[int]) -> tuple[int, int, int]:
    """Fifteens, pairs and runs for five cards, from their ranks alone.

    Called only while building :data:`RANK_TABLE`, so it is written for clarity
    rather than speed.
    """
    values = [RANK_VALUES[r] for r in ranks]
    n = len(values)

    # Fifteens: every non-empty subset summing to exactly 15 pays 2.
    fifteens = 0
    for mask in range(1, 1 << n):
        total = 0
        m = mask
        i = 0
        while m:
            if m & 1:
                total += values[i]
            m >>= 1
            i += 1
        if total == 15:
            fifteens += 2

    counts: dict[int, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1

    # Pairs: n of a kind is n*(n-1) points -- 2, 6, 12 for two, three, four.
    pairs = sum(c * (c - 1) for c in counts.values())

    # Runs: each maximal consecutive block of >=3 distinct ranks scores its
    # length times the product of the multiplicities.  That multiplier is what
    # turns a double run into 8 and a triple run into 15.
    runs = 0
    distinct = sorted(counts)
    i = 0
    while i < len(distinct):
        j = i
        while j + 1 < len(distinct) and distinct[j + 1] == distinct[j] + 1:
            j += 1
        length = j - i + 1
        if length >= 3:
            multiplier = 1
            for r in distinct[i : j + 1]:
                multiplier *= counts[r]
            runs += length * multiplier
        i = j + 1

    return fifteens, pairs, runs


def _build_rank_table() -> dict[tuple[int, ...], tuple[int, int, int]]:
    """Every legal five-card rank multiset mapped to its (fifteens, pairs, runs)."""
    table: dict[tuple[int, ...], tuple[int, int, int]] = {}
    for ranks in combinations_with_replacement(range(13), 5):
        # A five-of-a-kind cannot be dealt from one deck.
        if max(ranks.count(r) for r in set(ranks)) > 4:
            continue
        table[ranks] = _score_rank_multiset(ranks)
    return table


#: Sorted five-rank tuple -> (fifteens, pairs, runs).  6175 entries.
RANK_TABLE = _build_rank_table()

#: The same table pre-summed, for the fast path that only needs a total.
_RANK_TOTALS = {k: sum(v) for k, v in RANK_TABLE.items()}


def score_hand(hand: Sequence[int], starter: int, is_crib: bool = False) -> int:
    """Score a four-card hand against the starter.

    ``is_crib`` matters only for the flush: a crib scores a flush only when all
    five cards match suit, while a hand scores 4 for its own four.
    """
    a, b, c, d = hand
    total = _RANK_TOTALS[
        tuple(sorted((a >> 2, b >> 2, c >> 2, d >> 2, starter >> 2)))
    ]

    # Flush.
    suit = a & 3
    if (b & 3) == suit and (c & 3) == suit and (d & 3) == suit:
        if (starter & 3) == suit:
            total += 5
        elif not is_crib:
            total += 4

    # Nobs: a jack *in hand* matching the starter's suit.  A jack turned as the
    # starter is his heels, scored at the cut instead.
    starter_suit = starter & 3
    for card in hand:
        if (card >> 2) == JACK and (card & 3) == starter_suit:
            total += 1
            break

    return total


def score_hand_detail(
    hand: Sequence[int], starter: int, is_crib: bool = False
) -> HandScore:
    """Same as :func:`score_hand` but returning the per-category breakdown."""
    if len(hand) != 4:
        raise ValueError(f"expected a four-card hand, got {len(hand)} cards")

    ranks = tuple(sorted([c >> 2 for c in hand] + [starter >> 2]))
    fifteens, pairs, runs = RANK_TABLE[ranks]

    flush = 0
    suit = hand[0] & 3
    if all((c & 3) == suit for c in hand[1:]):
        if (starter & 3) == suit:
            flush = 5
        elif not is_crib:
            flush = 4

    nobs = 0
    starter_suit = starter & 3
    for card in hand:
        if (card >> 2) == JACK and (card & 3) == starter_suit:
            nobs = 1
            break

    return HandScore(fifteens=fifteens, pairs=pairs, runs=runs, flush=flush, nobs=nobs)


def score_play(seq: Sequence[int]) -> int:
    """Points scored by the card just added to ``seq``.

    ``seq`` is the cards played *since the last reset* -- never the whole round.
    Pairs and runs are read off the tail of that sequence, which is why a reset
    is a hard wall: cards from a previous sub-round are behind the wall and
    cannot combine with anything.
    """
    n = len(seq)
    if n == 0:
        return 0

    total = 0

    count = 0
    for c in seq:
        count += CARD_VALUES[c]
    if count == 15 or count == 31:
        total += 2

    # Pairs run backwards from the last card only.  In 5-5-3-3 the live pair is
    # the threes; the fives are buried.
    last_rank = seq[-1] >> 2
    same = 1
    while same < n and (seq[-1 - same] >> 2) == last_rank:
        same += 1
    if same > 1:
        total += same * (same - 1)

    # Runs: the longest suffix of length >= 3 whose ranks are distinct and span
    # exactly that length.  Cards may arrive out of order, so 3-4-2-5 is a run
    # of four, while 3-4-3-5 only qualifies on its last three cards.
    for length in range(n, 2, -1):
        ranks = {c >> 2 for c in seq[-length:]}
        if len(ranks) == length and max(ranks) - min(ranks) == length - 1:
            total += length
            break

    return total


def score_play_detail(seq: Sequence[int]) -> PlayScore:
    """Same as :func:`score_play` but returning the per-category breakdown."""
    n = len(seq)
    if n == 0:
        return PlayScore()

    count = sum(CARD_VALUES[c] for c in seq)
    fifteen = 2 if count == 15 else 0
    thirty_one = 2 if count == 31 else 0

    last_rank = seq[-1] >> 2
    same = 1
    while same < n and (seq[-1 - same] >> 2) == last_rank:
        same += 1
    pairs = same * (same - 1) if same > 1 else 0

    run = 0
    for length in range(n, 2, -1):
        ranks = {c >> 2 for c in seq[-length:]}
        if len(ranks) == length and max(ranks) - min(ranks) == length - 1:
            run = length
            break

    return PlayScore(fifteen=fifteen, thirty_one=thirty_one, pairs=pairs, run=run)
