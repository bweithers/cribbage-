"""Expected crib value for a two-card lay-away.

Choosing a discard means trading hand value against crib value, and the crib
term is the awkward half: its value depends on two cards you cannot see plus a
starter.  This module tabulates it.

The table is **exact under one stated assumption**: that the opponent's two crib
cards are a uniformly random pair from the rest of the deck.  Given a lay-away,
every opponent pair and every starter is enumerated -- no sampling, so no noise.
Real opponents are not uniform (the pone lays away defensively, the dealer
helpfully), so a strong agent would eventually want separate dealer and pone
tables conditioned on opponent policy.  This one table for both is the standard
first approximation and is documented as such rather than hidden.

The other approximation: the table is keyed only on the lay-away's ranks and
whether the two cards share a suit, so it ignores which cards are in the rest of
your hand.  Suitedness is kept because it is the only way a crib flush can
happen.

Keys are ``"{low_rank}_{high_rank}_{suited}"``.  169 entries: 91 unsuited pairs
(78 distinct-rank plus 13 same-rank) and 78 suited.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Optional

from .cards import NUM_CARDS
from .scoring import score_hand

__all__ = ["crib_ev", "compute_crib_table", "load_crib_table", "table_key", "DATA_FILE"]

DATA_FILE = "crib_ev.json"

_TABLE: Optional[dict[str, float]] = None


def table_key(card_a: int, card_b: int) -> str:
    """Canonical key for a lay-away: sorted ranks plus a suited flag."""
    ra, rb = card_a >> 2, card_b >> 2
    lo, hi = (ra, rb) if ra <= rb else (rb, ra)
    suited = 1 if (card_a & 3) == (card_b & 3) else 0
    return f"{lo}_{hi}_{suited}"


def compute_crib_table(progress=None) -> dict[str, float]:
    """Enumerate the exact expectation for every canonical lay-away.

    Roughly ten million hand evaluations, so this takes tens of seconds.  It is
    a build step, not something to call at import.
    """
    table: dict[str, float] = {}
    entries = []
    for lo in range(13):
        for hi in range(lo, 13):
            for suited in (0, 1):
                if suited and lo == hi:
                    continue  # a pair cannot share a suit
                entries.append((lo, hi, suited))

    for index, (lo, hi, suited) in enumerate(entries):
        mine = ((lo << 2) | 0, (hi << 2) | (0 if suited else 1))
        rest = [c for c in range(NUM_CARDS) if c not in mine]

        total = 0
        samples = 0
        for starter in rest:
            pool = [c for c in rest if c != starter]
            for theirs in combinations(pool, 2):
                crib = (mine[0], mine[1], theirs[0], theirs[1])
                total += score_hand(crib, starter, is_crib=True)
                samples += 1
        table[f"{lo}_{hi}_{suited}"] = total / samples
        if progress is not None:
            progress(index + 1, len(entries))

    return table


def _data_path() -> Path:
    return Path(__file__).parent / "data" / DATA_FILE


def load_crib_table() -> dict[str, float]:
    """The table, from the shipped data file if present, else computed once."""
    global _TABLE
    if _TABLE is None:
        path = _data_path()
        if path.exists():
            _TABLE = json.loads(path.read_text())
        else:  # pragma: no cover - only hit when the data file is missing
            _TABLE = compute_crib_table()
    return _TABLE


def crib_ev(card_a: int, card_b: int) -> float:
    """Expected points the crib is worth if ``card_a`` and ``card_b`` go into it."""
    return load_crib_table()[table_key(card_a, card_b)]
