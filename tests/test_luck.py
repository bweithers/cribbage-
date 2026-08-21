"""Cut-luck attribution: is the number well defined, and does it mean anything?"""

from __future__ import annotations

import random
import statistics

import pytest

from cribbage.arena import play_game
from cribbage.agents import HeuristicAgent, RandomAgent
from cribbage.cards import parse_card, parse_hand
from cribbage.engine import RoundRecord, new_game
from cribbage.luck import CutLuck, cut_luck, round_cut_luck


def finished_game(seed: int):
    state = new_game(seed=seed)
    rng = random.Random(seed)
    while not state.is_terminal():
        state.apply_action(rng.choice(state.legal_actions()))
    return state


# ----------------------------------------------------------------------
# Definition
# ----------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_net_cut_luck_is_exactly_zero_sum(seed):
    """One player's good cut is the other's bad cut, by construction."""
    state = finished_game(seed)
    assert cut_luck(state, 0).net == pytest.approx(-cut_luck(state, 1).net)


def test_expectation_covers_all_forty_cards_that_could_have_been_cut():
    state = finished_game(7)
    record = state.round_records[0]
    assert len(record.candidates) == 40
    assert record.starter in record.candidates
    dealt = set(record.dealt[0]) | set(record.dealt[1])
    assert len(dealt) == 12
    assert dealt.isdisjoint(record.candidates)


def test_the_twenty_nine_hand_is_mostly_deal_luck_not_cut_luck():
    """The pone holds three fives and the matching jack; the case five is cut.

    That is the 29 hand, the best in cribbage -- and the measure correctly
    declines to call it extreme *cut* luck.  A holding of three fives and a jack
    already averages 16.9 points against a random starter, so the case five is
    worth about twelve gross, and the dealer's own hand and crib both like a
    five too, which halves it again on net.  Being dealt that holding is the
    luck; the cut only finished it off.  Separating those two is the point.
    """
    pone_kept = parse_hand("5H 5D 5C JS")
    dealer_kept = parse_hand("2D 4H 6C 8S")
    crib = parse_hand("3D 7H 9C KS")
    record = RoundRecord(
        round_index=1, dealer=0,
        dealt=(tuple(dealer_kept + crib[:2]), tuple(pone_kept + crib[2:])),
        kept=(tuple(dealer_kept), tuple(pone_kept)),
        crib=tuple(crib),
        starter=parse_card("5S"),
    )
    row = round_cut_luck(record, player=1)

    assert row.actual_gross == 29, "the 29 hand"
    assert row.expected_gross > 15, (
        "the holding is already a monster before the cut: "
        f"expected {row.expected_gross:.1f}"
    )
    assert 3.0 < row.net < 10.0, f"net cut luck was {row.net:.1f}"
    assert 0.5 < row.net_z < 1.5, (
        f"a good cut, not a freak one: z = {row.net_z:.2f}"
    )


def test_a_shared_windfall_nets_out_smaller_than_it_looks_gross():
    """A cut that helps both sides is not luck for either.

    Both players keep four ten-value cards, so a five is worth four fifteens to
    each of them.  Gross it looks like a windfall; on net most of it cancels,
    which is exactly why the net view is the headline number.
    """
    record = RoundRecord(
        round_index=1, dealer=0,
        dealt=(tuple(parse_hand("TS JH QC KD 2S 3S")),
               tuple(parse_hand("TH JC QD KS 2H 3H"))),
        kept=(tuple(parse_hand("TS JH QC KD")), tuple(parse_hand("TH JC QD KS"))),
        crib=tuple(parse_hand("2S 3S 2H 3H")),
        starter=parse_card("5D"),
    )
    row = round_cut_luck(record, player=1)

    assert row.gross > 4.0, f"a five is a gross windfall here: {row.gross:.1f}"
    assert 0 < row.net < row.gross, (
        f"net {row.net:.1f} should survive but shrink against gross "
        f"{row.gross:.1f}, since the cut fed both sides"
    )


def test_a_five_is_not_automatically_a_good_cut():
    """Whether a card is lucky depends on the hand it lands beside.

    A five completes the run for 6-7-8-9, but adds no fifteens, and that holding
    does better with most other cards -- so this five is *below* average.
    """
    record = RoundRecord(
        round_index=1, dealer=0,
        dealt=(tuple(parse_hand("2D 4H TC KS 3D 5H")),
               tuple(parse_hand("6S 7S 8C 9C 9S QC"))),
        kept=(tuple(parse_hand("2D 4H TC KS")), tuple(parse_hand("6S 7S 8C 9C"))),
        crib=tuple(parse_hand("3D 5H 9S QC")),
        starter=parse_card("5D"),
    )
    row = round_cut_luck(record, player=1)
    assert row.gross < 0, f"the five was worth {row.gross:+.1f} gross to 6-7-8-9"


def test_rounds_cut_short_before_the_show_are_excluded_and_counted():
    for seed in range(12):
        state = finished_game(seed)
        luck = cut_luck(state, 0)
        assert luck.skipped >= 0
        assert len(luck.rounds) + luck.skipped == len(state.round_records)
        counted = {row.round_index for row in luck.rounds}
        scored = {e.round for e in state.events if e.kind == "crib"}
        assert counted == scored


def test_empty_game_is_handled():
    empty = CutLuck(player=0, rounds=())
    assert empty.net == 0 and empty.net_sd == 0 and empty.net_z == 0
    assert "par" in empty.describe()


# ----------------------------------------------------------------------
# Calibration and meaning
# ----------------------------------------------------------------------


def test_the_measure_is_unbiased_and_its_z_score_is_calibrated():
    """Mean zero by symmetry, and unit variance if the arithmetic is right.

    The z-score check is the sharp one: it only comes out at 1.0 if the
    per-round variances are correct *and* summing them across rounds is valid.
    """
    nets, zs = [], []
    rng = random.Random(23)
    for game in range(400):
        state = play_game([HeuristicAgent(), HeuristicAgent()],
                          seed=rng.randrange(1 << 30), dealer=game % 2)
        luck = cut_luck(state, 0)
        nets.append(luck.net)
        zs.append(luck.net_z)

    assert abs(statistics.mean(nets)) < 2.0, "should be centred on zero"
    assert 0.85 < statistics.pstdev(zs) < 1.15, (
        f"z scores had sd {statistics.pstdev(zs):.3f}, expected about 1.0"
    )


def test_better_cuts_predict_winning():
    """If cut luck did not track outcomes it would be decoration."""
    nets, wins = [], []
    rng = random.Random(29)
    for game in range(400):
        state = play_game([HeuristicAgent(), RandomAgent(seed=game)],
                          seed=rng.randrange(1 << 30), dealer=game % 2)
        nets.append(cut_luck(state, 0).net)
        wins.append(1 if state.result().winner == 0 else 0)

    order = sorted(range(len(nets)), key=lambda i: nets[i])
    third = len(order) // 3
    worst = statistics.mean(wins[i] for i in order[:third])
    best = statistics.mean(wins[i] for i in order[-third:])
    assert best > worst, f"best-third won {best:.1%}, worst-third {worst:.1%}"
