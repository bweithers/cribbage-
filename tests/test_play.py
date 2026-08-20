"""score_play: known sequences plus a differential fuzz against a naive scorer."""

from __future__ import annotations

import random

import pytest

from cribbage.cards import CARD_VALUES, parse_hand
from cribbage.scoring import score_play, score_play_detail

from .reference import naive_score_play


def played(seq: str) -> int:
    return score_play(parse_hand(seq))


@pytest.mark.parametrize(
    "seq,expected,note",
    [
        ("", 0, "nothing played"),
        ("5S", 0, "a lone card can never reach fifteen"),
        ("7S 8H", 2, "fifteen with two cards"),
        ("2S 3H TC", 2, "fifteen with three cards"),
        ("5S 7H 7D", 2, "a pair off the tail"),
        ("7S 7H 7D", 6, "three of a kind"),
        ("7S 7H 7D 7C", 12, "four of a kind"),
        ("5S 5H 3D 3C", 2, "pairs come off the tail: the fives are buried"),
        ("3S 4H 5D", 3, "a plain run of three"),
        ("4S 5H 6D", 5, "run of three that also makes fifteen"),
        ("3S 4H 2D 5C", 4, "a run of four that arrived out of order"),
        ("3S 4H 3D 5C", 5, "duplicate breaks the long run; 4-3-5 still runs, and it is fifteen"),
        ("3S 4H 5D 6C 7S", 5, "run of five"),
        ("JS QH KD", 3, "runs use rank: J-Q-K runs even though it counts 30"),
        ("KS KH 6D 5C", 2, "exactly thirty-one; the buried kings do not pair again"),
        ("7S 8H 8D 8C", 8, "thirty-one and three of a kind together"),
    ],
)
def test_known_sequences(seq, expected, note):
    assert played(seq) == expected, note


def test_jqk_runs_but_counts_thirty():
    seq = parse_hand("JS QH KD")
    assert sum(CARD_VALUES[c] for c in seq) == 30
    assert score_play_detail(seq).run == 3


def test_detail_sums_to_total():
    detail = score_play_detail(parse_hand("7S 8H 8D 8C"))
    assert (detail.thirty_one, detail.pairs) == (2, 6)
    assert detail.total == score_play(parse_hand("7S 8H 8D 8C"))


def random_sequence(rng: random.Random) -> list[int]:
    """A legal play sequence: distinct cards, running count never over 31."""
    deck = list(range(52))
    rng.shuffle(deck)
    seq: list[int] = []
    count = 0
    for card in deck:
        if len(seq) == 8 or count + CARD_VALUES[card] > 31:
            continue
        seq.append(card)
        count += CARD_VALUES[card]
        if len(seq) == 8:
            break
    return seq


def test_matches_naive_scorer_on_random_sequences():
    rng = random.Random(20240820)
    for _ in range(20000):
        seq = random_sequence(rng)
        for length in range(1, len(seq) + 1):
            prefix = seq[:length]
            assert score_play(prefix) == naive_score_play(prefix), prefix
