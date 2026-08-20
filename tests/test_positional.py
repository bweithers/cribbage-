"""The position-aware agent: that it reduces to the baseline, and that each
feature moves decisions in the direction it claims to.

The first test is the one the whole ablation experiment rests on -- if
``PositionalAgent`` with every feature disabled did not reproduce
``HeuristicAgent`` exactly, no measured difference could be attributed to the
feature under test.
"""

from __future__ import annotations

import random
from itertools import combinations

import pytest

from cribbage.agents import HeuristicAgent, PositionalAgent
from cribbage.cards import parse_card, parse_hand
from cribbage.engine import Phase, new_game
from cribbage.engine import InfoState

OFF = dict(endgame=False, finish=False)


def discard_view(hand, is_dealer: bool, my_score: int = 0, opp_score: int = 0) -> InfoState:
    cards = tuple(sorted(hand))
    return InfoState(
        player=0, phase=Phase.DISCARD, dealer=0 if is_dealer else 1, round_index=1,
        target=121, my_score=my_score, opp_score=opp_score, hand=cards,
        my_discards=(), starter=None, count=0, seq=(), play_order=(), my_played=(),
        opp_played=(), opp_hand_size=6, legal=tuple(combinations(cards, 2)),
    )


def play_view(hand, count: int, seq: str, my_score: int = 0, opp_score: int = 0) -> InfoState:
    cards = tuple(sorted(hand))
    playable = tuple(c for c in cards if count + min((c >> 2) + 1, 10) <= 31)
    return InfoState(
        player=0, phase=Phase.PLAY, dealer=0, round_index=1, target=121,
        my_score=my_score, opp_score=opp_score, hand=cards, my_discards=(),
        starter=parse_card("2D"), count=count,
        seq=tuple(parse_hand(seq)) if seq else (), play_order=(), my_played=(),
        opp_played=(), opp_hand_size=4, legal=playable,
    )


# ----------------------------------------------------------------------


def test_every_feature_off_reproduces_the_baseline_exactly():
    """The control for the ablation experiment.

    Any behavioural difference between these two would contaminate every
    measurement made against the baseline.
    """
    baseline = HeuristicAgent()
    control = PositionalAgent(
        endgame=False, finish=False,
        stance_variance=0.0, stance_crib=0.0, stance_peg=0.0,
        crib_defense=0.0, five_penalty=0.0,
    )
    decisions = 0
    for seed in range(12):
        state = new_game(seed=seed)
        while not state.is_terminal():
            player = state.current_player
            info = state.information_state(player)
            assert control.act(info) == baseline.act(info)
            decisions += 1
            state.apply_action(baseline.act(info))
    assert decisions > 800, "expected a substantial number of compared decisions"


def test_stance_reads_the_score():
    agent = PositionalAgent()
    ahead = agent.stance(discard_view(parse_hand("AS 2H 3D 4C 5S KH"), False, 100, 40))
    behind = agent.stance(discard_view(parse_hand("AS 2H 3D 4C 5S KH"), False, 40, 100))
    level = agent.stance(discard_view(parse_hand("AS 2H 3D 4C 5S KH"), False, 60, 60))
    assert ahead > 0.9 and behind < -0.9
    assert behind < level < ahead
    assert -1.0 <= behind and ahead <= 1.0


def test_holding_the_deal_counts_as_being_ahead():
    agent = PositionalAgent()
    hand = parse_hand("AS 2H 3D 4C 5S KH")
    as_dealer = agent.stance(discard_view(hand, True, 60, 60))
    as_pone = agent.stance(discard_view(hand, False, 60, 60))
    assert as_dealer > as_pone


def test_ahead_keeps_the_safe_hand_and_behind_reaches_for_variance():
    """Same six cards, opposite ends of the board, opposite lay-aways."""
    hand = parse_hand("AC 2S 3H 6S 7H JH")
    agent = PositionalAgent(**OFF, stance_variance=1.2)
    options = {o.pair: o for o in agent.discard_options(discard_view(hand, False))}

    ahead = options[tuple(agent.discard(discard_view(hand, False, 100, 60)))]
    behind = options[tuple(agent.discard(discard_view(hand, False, 60, 100)))]

    assert ahead.pair != behind.pair
    assert ahead.stdev < behind.stdev, (
        f"ahead kept sd {ahead.stdev:.2f}, behind kept sd {behind.stdev:.2f}"
    )


def test_endgame_trades_expected_points_for_a_better_chance_of_going_out():
    """As pone needing 14, take the lay-away that reaches it more often.

    The baseline keeps the higher-scoring hand on average; the endgame agent
    gives up half a point of expectation to roughly double the chance of
    actually finishing the game, which is the only thing that counts here.

    Uses the unsmoothed objective, which thresholds on expected pegging.  The
    smoothed one folds over the pegging distribution instead and is less willing
    to make this particular trade -- see the tests below.
    """
    hand = parse_hand("AD 2S 2H 4H 7D 7C")
    info = discard_view(hand, False, my_score=121 - 14)
    baseline = HeuristicAgent()
    endgame = PositionalAgent(
        **{**OFF, "endgame": True, "endgame_smoothing": False}
    )

    options = {o.pair: o for o in baseline.discard_options(info)}
    chosen_base = options[tuple(baseline.discard(info))]
    chosen_end = options[tuple(endgame.discard(info))]

    assert chosen_base.pair != chosen_end.pair
    needed = endgame.points_needed_from_hand(chosen_end, info)
    assert chosen_end.probability_at_least(needed) > chosen_base.probability_at_least(needed)
    assert chosen_end.mean < chosen_base.mean, "the trade should cost expected points"


def test_smoothing_folds_over_the_pegging_distribution():
    """The smoothed probability is exactly the distribution-weighted sum."""
    from cribbage.agents.positional import PEG_DISTRIBUTION_PONE

    hand = parse_hand("AD 2S 2H 4H 7D 7C")
    info = discard_view(hand, False, my_score=121 - 14)
    agent = PositionalAgent(**{**OFF, "endgame": True})
    option = agent.discard_options(info)[0]

    expected = sum(
        weight * option.probability_at_least(14 - pegged)
        for pegged, weight in PEG_DISTRIBUTION_PONE
    )
    assert agent.probability_of_going_out(option, info) == pytest.approx(expected)
    assert sum(weight for _, weight in PEG_DISTRIBUTION_PONE) == pytest.approx(1.0)


def test_smoothed_probability_is_bounded_by_the_thresholds_it_averages():
    """A weighted average cannot fall outside the values it averages over."""
    from cribbage.agents.positional import PEG_DISTRIBUTION_PONE

    hand = parse_hand("AD 2S 2H 4H 7D 7C")
    info = discard_view(hand, False, my_score=121 - 14)
    agent = PositionalAgent(**{**OFF, "endgame": True})
    for option in agent.discard_options(info):
        parts = [
            option.probability_at_least(14 - pegged)
            for pegged, _ in PEG_DISTRIBUTION_PONE
        ]
        smoothed = agent.probability_of_going_out(option, info)
        assert min(parts) <= smoothed <= max(parts)


def test_going_out_gets_less_likely_the_more_points_are_needed():
    """Monotonicity: needing more points can never raise the chance of getting them."""
    agent = PositionalAgent(**{**OFF, "endgame": True})
    hand = parse_hand("AD 2S 2H 4H 7D 7C")
    option = agent.discard_options(discard_view(hand, False))[0]

    previous = 1.0
    for needed in range(2, 32):
        info = discard_view(hand, False, my_score=121 - needed)
        current = agent.probability_of_going_out(option, info)
        assert current <= previous + 1e-12, f"rose at {needed} points needed"
        previous = current
    assert previous == 0.0, "thirty-one points cannot come off one hand"


def test_endgame_stops_discriminating_when_the_game_is_not_close():
    """With 121 to go, nothing can finish the deal, so points decide again."""
    hand = parse_hand("AD 2S 2H 4H 7D 7C")
    info = discard_view(hand, False, my_score=0)
    assert (
        PositionalAgent(**{**OFF, "endgame": True}).discard(info)
        == HeuristicAgent().discard(info)
    )


def test_crib_defense_lays_away_less_valuable_cards_as_pone():
    """Turning up the defence should reduce what gets handed to the dealer."""
    agent_soft = PositionalAgent(**OFF)
    agent_hard = PositionalAgent(**{**OFF, "crib_defense": 3.0})
    rng = random.Random(11)
    softer, harder = 0.0, 0.0
    for _ in range(80):
        hand = rng.sample(range(52), 6)
        info = discard_view(hand, False)
        options = {o.pair: o for o in agent_soft.discard_options(info)}
        softer += options[tuple(agent_soft.discard(info))].crib
        harder += options[tuple(agent_hard.discard(info))].crib
    assert harder < softer, (
        f"defended lay-aways averaged {harder / 80:.2f} crib points, "
        f"undefended {softer / 80:.2f}"
    )


def test_crib_defense_does_nothing_when_dealing():
    """It is your own crib; there is nothing to defend."""
    hand = parse_hand("AS 2H 3D 4C 5S KH")
    info = discard_view(hand, True)
    assert (
        PositionalAgent(**{**OFF, "crib_defense": 3.0}).discard(info)
        == PositionalAgent(**OFF).discard(info)
    )


def test_finish_takes_a_card_that_wins_outright():
    agent = PositionalAgent(finish=True)
    # Two points from home, with a five that makes fifteen for exactly two.
    info = play_view(parse_hand("5S 9D KC 3H"), count=10, seq="TC", my_score=119)
    assert agent.play(info) == parse_card("5S")


def test_stance_peg_fears_the_reply_more_when_ahead():
    agent = PositionalAgent(**{**OFF, "stance_peg": 0.8})
    hand = parse_hand("5S 7H 9D KC")
    ahead = agent.effective_risk(play_view(hand, 0, "", my_score=100, opp_score=40))
    behind = agent.effective_risk(play_view(hand, 0, "", my_score=40, opp_score=100))
    assert ahead > agent.risk_weight > behind >= 0.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_positional_only_returns_legal_actions(seed):
    agent = PositionalAgent(stance_variance=0.3, stance_crib=0.4, stance_peg=0.5)
    state = new_game(seed=seed)
    rng = random.Random(seed)
    while not state.is_terminal():
        player = state.current_player
        info = state.information_state(player)
        action = agent.act(info) if player == 0 else rng.choice(info.legal)
        assert action in info.legal
        state.apply_action(action)
