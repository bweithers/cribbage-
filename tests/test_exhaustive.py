"""The exhaustive scoring property, as an opt-in test.

Run with ``pytest --runslow``.  See ``scripts/exhaustive_scoring_check.py`` for
the standalone version with progress output.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import pytest

from cribbage.cards import parse_card, parse_hand
from cribbage.scoring import score_hand

IMPOSSIBLE = (19, 25, 26, 27)


@pytest.mark.slow
@pytest.mark.parametrize("is_crib", [False, True])
def test_no_hand_can_total_nineteen_twentyfive_twentysix_or_twentyseven(is_crib):
    histogram: Counter[int] = Counter()
    for five in combinations(range(52), 5):
        for index in range(5):
            hand = five[:index] + five[index + 1 :]
            histogram[score_hand(hand, five[index], is_crib=is_crib)] += 1

    assert sum(histogram.values()) == 2_598_960 * 5
    for total in IMPOSSIBLE:
        assert histogram[total] == 0, f"{histogram[total]} hands scored {total}"
    assert max(histogram) == 29
    assert min(histogram) == 0


def test_the_twenty_nine_hand_is_the_only_way_to_reach_it():
    """Cheap corollary of the exhaustive check: 29 needs three fives and the right jack."""
    best = []
    for suit in range(4):
        jack = (10 << 2) | suit
        starter = (4 << 2) | suit
        fives = [(4 << 2) | s for s in range(4) if s != suit]
        best.append(score_hand([jack] + fives, starter))
    assert best == [29, 29, 29, 29]
    # Swap the jack's suit and it drops to 28: nobs is the difference.
    assert score_hand(parse_hand("JH 5H 5D 5C"), parse_card("5S")) == 28
