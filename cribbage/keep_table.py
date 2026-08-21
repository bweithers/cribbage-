"""Expected show value of a four-card keep, by rank.

Used as a cheap plausibility weight when guessing what an opponent is holding.
Sampling their hand uniformly from the unseen cards is wrong in a specific way:
they did not receive four random cards, they received six and *chose* four.  A
holding of 5-5-6-9 is far likelier to be in front of them than 2-7-9-K, because
nobody keeps the second.

So weight each sampled holding by how good a keep it is.  This table is the
weight: for every four-card rank multiset, the mean hand score over all starters.
Exact and cheap -- 1820 multisets against 13 starter ranks is a few hundred
thousand dictionary lookups -- so it is computed on first use rather than shipped.

Ranks only, so the flush is ignored.  As a *relative* plausibility weight that
hardly matters, and paying for suits would cost far more than it bought.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .cards import RANK_VALUES
from .peg_table import all_keys, cards_for_ranks, table_key
from .scoring import score_hand

__all__ = ["keep_value", "load_keep_table"]

_TABLE: Optional[dict[str, float]] = None


def _build() -> dict[str, float]:
    table: dict[str, float] = {}
    for ranks in all_keys():
        keep = cards_for_ranks(ranks)
        used = set(keep)
        # One starter per rank is enough: suits only matter for the flush and
        # for nobs, neither of which this weight is trying to capture.
        total = 0.0
        weight = 0
        for rank in range(13):
            for suit in range(4):
                starter = (rank << 2) | suit
                if starter not in used:
                    total += score_hand(keep, starter)
                    weight += 1
                    break
        table[table_key(ranks)] = total / weight
    return table


def load_keep_table() -> dict[str, float]:
    global _TABLE
    if _TABLE is None:
        _TABLE = _build()
    return _TABLE


def keep_value(ranks: Sequence[int]) -> float:
    """Mean show score of a keep with these ranks, over the possible starters."""
    return load_keep_table()[table_key(ranks)]


def hand_total(ranks: Sequence[int]) -> int:
    """Total pip value of the ranks, for quick sanity checks."""
    return sum(RANK_VALUES[rank] for rank in ranks)
