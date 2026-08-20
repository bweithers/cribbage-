"""Hand scoring: known fixtures plus a differential fuzz against a naive scorer."""

from __future__ import annotations

import random

import pytest

from cribbage.cards import parse_card, parse_hand
from cribbage.scoring import RANK_TABLE, score_hand, score_hand_detail

from .reference import naive_score_hand


def score(hand: str, starter: str, is_crib: bool = False) -> int:
    return score_hand(parse_hand(hand), parse_card(starter), is_crib)


def test_rank_table_is_complete():
    # C(17,5) five-card multisets over 13 ranks, less the 13 five-of-a-kinds.
    assert len(RANK_TABLE) == 6175


@pytest.mark.parametrize(
    "hand,starter,expected,note",
    [
        ("JS 5H 5D 5C", "5S", 29, "the 29 hand: 16 fifteens, 12 pairs, nobs"),
        ("JH 5H 5D 5C", "5S", 28, "same shape, but the jack misses the starter's suit"),
        ("5S 5H 5D 5C", "KS", 28, "four fives and a ten-value starter"),
        ("4S 5H 5D 6C", "6S", 24, "double-double run: 12 for runs, 8 fifteens, 4 pairs"),
        ("2S 3H 4D 4C", "4S", 17, "triple run of three: 9 runs, 6 pairs, one fifteen"),
        ("4S 5H 5D 6C", "9S", 14, "double run of three: 6 runs, 6 fifteens, 2 pairs"),
        ("AS 2H 3D 4C", "5S", 7, "run of five plus the single fifteen it makes"),
        ("AS 2H 3D 5C", "KS", 7, "run of three plus K-5 and K-3-2"),
        ("AS 2H 4D 6C", "8S", 4, "two fifteens and nothing else"),
        ("AS 8H 9D TC", "KS", 3, "a bare run of three; no subset reaches fifteen"),
        ("2S 4H 6D 8C", "TS", 0, "every card even, so no subset can sum to an odd 15"),
    ],
)
def test_known_hands(hand, starter, expected, note):
    assert score(hand, starter) == expected, note


def test_flush_in_hand_is_four_and_five_with_the_starter():
    # A-4-6-8 makes two fifteens (A-4-T and A-6-8) whatever the starter's suit.
    assert score("AS 4S 6S 8S", "TH") == 4 + 4  # four-card flush
    assert score("AS 4S 6S 8S", "TS") == 4 + 5  # starter matches: five-card flush


def test_crib_needs_all_five_for_a_flush():
    # Identical cards: four fifteen-points either way, but the flush is worth 4
    # in the hand and nothing at all in the crib.
    assert score("2S 4S 6S 9S", "TH") == 4 + 4
    assert score("2S 4S 6S 9S", "TH", is_crib=True) == 4
    # A true five-card flush still counts in the crib.
    assert score("2S 4S 6S 9S", "TS", is_crib=True) == 4 + 5


def test_nobs_needs_a_matching_suit_and_is_not_his_heels():
    assert score_hand_detail(parse_hand("JS 2H 4D 6C"), parse_card("8S")).nobs == 1
    assert score_hand_detail(parse_hand("JS 2H 4D 6C"), parse_card("8H")).nobs == 0
    # A jack turned as the starter is his heels, paid at the cut, never nobs.
    assert score_hand_detail(parse_hand("2S 4H 6D 8C"), parse_card("JS")).nobs == 0


def test_detail_sums_to_total():
    detail = score_hand_detail(parse_hand("JS 5H 5D 5C"), parse_card("5S"))
    assert (detail.fifteens, detail.pairs, detail.nobs) == (16, 12, 1)
    assert detail.total == 29
    assert detail.total == score("JS 5H 5D 5C", "5S")


@pytest.mark.parametrize("is_crib", [False, True])
def test_matches_naive_scorer_on_random_hands(is_crib):
    rng = random.Random(20240820 + is_crib)
    for _ in range(20000):
        cards = rng.sample(range(52), 5)
        hand, starter = cards[:4], cards[4]
        assert score_hand(hand, starter, is_crib) == naive_score_hand(
            hand, starter, is_crib
        ), (hand, starter, is_crib)
