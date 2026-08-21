"""The pegging-value table, and the agent that uses it.

The table is sampled rather than exact, so it cannot be asserted entry by entry.
What can be asserted is that it is complete, that it is shaped the way cribbage
says it should be, and that the agent reduces to its baseline when the new term
is switched off -- the control the measurement depends on.
"""

from __future__ import annotations

import random
from itertools import combinations

import pytest

from cribbage.agents import HeuristicAgent, PeggingAwareAgent
from cribbage.cards import parse_hand, rank_of
from cribbage.engine import InfoState, Phase, new_game
from cribbage.peg_table import (
    all_keys, cards_for_ranks, load_peg_table, peg_ev, table_key,
)


def discard_view(hand, is_dealer: bool) -> InfoState:
    cards = tuple(sorted(hand))
    return InfoState(
        player=0, phase=Phase.DISCARD, dealer=0 if is_dealer else 1, round_index=1,
        target=121, my_score=0, opp_score=0, hand=cards, my_discards=(), starter=None,
        count=0, seq=(), play_order=(), my_played=(), opp_played=(), opp_hand_size=6,
        legal=tuple(combinations(cards, 2)),
    )


def ranks_of(text: str) -> list[int]:
    return sorted(rank_of(card) for card in parse_hand(text))


# ----------------------------------------------------------------------
# The table
# ----------------------------------------------------------------------


def test_every_four_card_rank_multiset_is_covered():
    keys = all_keys()
    assert len(keys) == 1820, "C(16,4) four-card multisets over 13 ranks"
    table = load_peg_table()
    assert len(table) == len(keys)
    for ranks in keys:
        assert table_key(ranks) in table
        assert len(table[table_key(ranks)]) == 2, "one value per seat"


def test_table_key_ignores_the_order_cards_are_given_in():
    assert table_key([12, 0, 5, 5]) == table_key([0, 5, 5, 12])


def test_cards_for_ranks_builds_a_real_holding():
    cards = cards_for_ranks([4, 4, 4, 10])
    assert len(set(cards)) == 4, "four distinct cards"
    assert sorted(rank_of(card) for card in cards) == [4, 4, 4, 10]


def test_dealing_is_worth_more_than_not_for_essentially_every_keep():
    """The dealer plays the last card, so the same cards peg better in their hand."""
    table = load_peg_table()
    better = sum(1 for pone, dealer in table.values() if dealer > pone)
    assert better / len(table) > 0.99, f"only {better}/{len(table)}"

    mean_pone = sum(value[0] for value in table.values()) / len(table)
    mean_dealer = sum(value[1] for value in table.values()) / len(table)
    assert 1.5 < mean_dealer - mean_pone < 4.0, (
        f"dealer advantage came out at {mean_dealer - mean_pone:.2f} points"
    )


def test_the_worst_keep_depends_on_which_seat_you_are_in():
    """A finding worth pinning down, because folklore blurs it.

    The pone *leads*, so a fistful of fives is ruinous -- every sub-round starts
    by handing over fifteen-for-two, and 5-5-5-5 is the single worst pone keep of
    all 1820.  The dealer *responds*, so the ruinous holding is four ten-value
    cards: once the count passes 21 there is nothing legal left to answer with.
    T-J-Q-K manages to be near the bottom in both seats at once, which is why it
    is the example everyone reaches for.
    """
    table = load_peg_table()
    by_pone = sorted(table, key=lambda key: table[key][0])
    by_dealer = sorted(table, key=lambda key: table[key][1])
    total = len(table)

    def pone_rank(hand: str) -> int:
        return by_pone.index(table_key(ranks_of(hand)))

    def dealer_rank(hand: str) -> int:
        return by_dealer.index(table_key(ranks_of(hand)))

    # Bad wherever you sit.
    assert pone_rank("TS JH QD KC") < total * 0.01
    assert dealer_rank("TS JH QD KC") < total * 0.01

    # Leading is the pone's problem: fives are poison in that seat only.
    assert pone_rank("5S 5H 5D 5C") < total * 0.01
    assert dealer_rank("5S 5H 5D 5C") > pone_rank("5S 5H 5D 5C") * 20

    # Answering is the dealer's problem: ten-value hands are poison in theirs.
    for hand in ("8S 8H 8D 8C", "JS JH JD JC", "TS TH TD TC"):
        assert dealer_rank(hand) < total * 0.02, f"{hand} ranked {dealer_rank(hand)}"

    # Low connected cards peg well in either seat.
    assert pone_rank("AS 2H 3D 4C") > total * 0.5
    assert dealer_rank("AS 2H 3D 4C") > total * 0.9


def test_peg_values_stay_in_a_believable_range():
    """A whole round's pegging is a handful of points, so a differential cannot
    be enormous."""
    table = load_peg_table()
    for key, (pone, dealer) in table.items():
        assert -8 < pone < 8 and -8 < dealer < 8, f"{key} -> {pone}, {dealer}"


# ----------------------------------------------------------------------
# The agent
# ----------------------------------------------------------------------


def test_zero_weight_reproduces_the_baseline_exactly():
    """The control the whole comparison rests on."""
    baseline = HeuristicAgent()
    control = PeggingAwareAgent(peg_weight=0.0)
    decisions = 0
    for seed in range(8):
        state = new_game(seed=seed)
        while not state.is_terminal():
            player = state.current_player
            info = state.information_state(player)
            assert control.act(info) == baseline.act(info)
            decisions += 1
            state.apply_action(baseline.act(info))
    assert decisions > 500


def test_the_pegging_term_changes_a_substantial_share_of_discards():
    """Unlike the position features, this one actually engages.

    Those moved one to three percent of discards, which is why their null
    results were uninformative. This should be an order of magnitude larger.
    """
    baseline, pegging = HeuristicAgent(), PeggingAwareAgent(peg_weight=1.0)
    rng = random.Random(3)
    differ = total = 0
    for _ in range(200):
        hand = rng.sample(range(52), 6)
        for is_dealer in (True, False):
            info = discard_view(hand, is_dealer)
            total += 1
            differ += baseline.discard(info) != pegging.discard(info)
    assert differ / total > 0.08, f"only {differ / total:.1%} of discards changed"


def test_it_will_pay_hand_points_to_escape_a_dreadful_pegging_hand():
    """Holding T-J-Q-K plus two low cards, the baseline keeps the run of four."""
    hand = parse_hand("3D 6S TD JD QS KH")
    info = discard_view(hand, False)
    baseline_keep = set(hand) - set(HeuristicAgent().discard(info))
    pegging_keep = set(hand) - set(PeggingAwareAgent(peg_weight=1.0).discard(info))

    assert baseline_keep == set(parse_hand("TD JD QS KH"))
    assert pegging_keep != baseline_keep
    agent = PeggingAwareAgent(peg_weight=1.0)
    assert peg_ev([rank_of(c) for c in pegging_keep], False) > peg_ev(
        [rank_of(c) for c in baseline_keep], False
    )
    assert agent.pegging_value  # the hook exists for subclasses to override


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_agent_only_returns_legal_actions(seed):
    agent = PeggingAwareAgent()
    rng = random.Random(seed)
    state = new_game(seed=seed)
    while not state.is_terminal():
        player = state.current_player
        info = state.information_state(player)
        action = agent.act(info) if player == 0 else rng.choice(info.legal)
        assert action in info.legal
        state.apply_action(action)


def test_the_best_reply_model_is_off_by_default_but_still_works():
    """Kept, defaulted off, because it measured null -- see the class docstring.

    It is more principled than the average-reply model it replaces, and it does
    produce sharply higher (more realistic) estimates. It just does not change
    enough decisions to show up in a win rate.
    """
    from cribbage.cards import parse_hand

    assert PeggingAwareAgent().best_reply is False

    average = PeggingAwareAgent(best_reply=False)
    best = PeggingAwareAgent(best_reply=True)
    after = parse_hand("4H")
    unseen = [card for card in range(52) if card not in set(after)]

    mean_cost, mean_stuck = average.reply_cost(after, unseen, 27, 4, len(unseen))
    best_cost, best_stuck = best.reply_cost(after, unseen, 27, 4, len(unseen))

    assert best_cost > mean_cost, (
        "assuming the opponent finds their best card must cost more than "
        f"assuming an average one: {best_cost:.3f} vs {mean_cost:.3f}"
    )
    assert mean_stuck == pytest.approx(best_stuck), (
        "both models agree on whether the opponent can play at all"
    )


def test_best_reply_is_a_proper_expectation_over_the_maximum():
    """With a single card held, the maximum is just the card, so the two models
    must agree exactly."""
    from cribbage.cards import parse_hand

    after = parse_hand("4H")
    unseen = [card for card in range(52) if card not in set(after)]
    average = PeggingAwareAgent(best_reply=False).reply_cost(
        after, unseen, 27, 1, len(unseen)
    )
    best = PeggingAwareAgent(best_reply=True).reply_cost(
        after, unseen, 27, 1, len(unseen)
    )
    assert best[0] == pytest.approx(average[0])
    assert best[1] == pytest.approx(average[1])


# ----------------------------------------------------------------------
# Search depth, and the controls that measure what the play is worth
# ----------------------------------------------------------------------


def test_expected_max_matches_the_average_when_only_one_card_is_held():
    """With a single card the maximum *is* the card, so the two agree exactly."""
    from cribbage.agents.pegging import expected_max

    values = [0.0, 2.0, 2.0, 6.0]
    got, stuck = expected_max(values, unplayable=0, held=1, pool=4)
    assert got == pytest.approx(sum(values) / len(values))
    assert stuck == 0.0


def test_expected_max_is_bounded_by_the_values_it_ranges_over():
    from cribbage.agents.pegging import expected_max

    values = [-3.0, 0.0, 1.0, 7.0]
    for held in (1, 2, 3, 4):
        got, _ = expected_max(values, unplayable=0, held=held, pool=len(values))
        assert min(values) <= got <= max(values)


def test_expected_max_rises_with_the_number_of_cards_held():
    """More cards to choose from can only help the chooser."""
    from cribbage.agents.pegging import expected_max

    values = [0.0, 1.0, 2.0, 5.0, 9.0, 2.0]
    seen = [expected_max(values, 0, held, len(values))[0] for held in range(1, 6)]
    assert seen == sorted(seen)


def test_being_stuck_is_certain_when_nothing_is_playable():
    from cribbage.agents.pegging import expected_max

    got, stuck = expected_max([], unplayable=5, held=2, pool=5)
    assert got == 0.0 and stuck == pytest.approx(1.0)


def test_depth_one_reproduces_the_baseline_play():
    baseline = HeuristicAgent()
    shallow = PeggingAwareAgent(peg_weight=0.0, best_reply=False, play_depth=1)
    checked = 0
    for seed in range(6):
        state = new_game(seed=seed)
        while not state.is_terminal():
            player = state.current_player
            info = state.information_state(player)
            if info.phase is Phase.PLAY:
                assert shallow.play(info) == baseline.play(info)
                checked += 1
            state.apply_action(baseline.act(info))
    assert checked > 200


def test_two_ply_changes_a_meaningful_share_of_play_decisions():
    """Unlike the best-reply model, searching deeper actually moves decisions."""
    baseline = HeuristicAgent()
    deep = PeggingAwareAgent(peg_weight=0.0, play_depth=2)
    differ = total = 0
    for seed in range(12):
        state = new_game(seed=seed)
        while not state.is_terminal():
            player = state.current_player
            info = state.information_state(player)
            if info.phase is Phase.PLAY and len(info.legal) > 1:
                total += 1
                differ += deep.play(info) != baseline.play(info)
            state.apply_action(baseline.act(info))
    assert differ / total > 0.08, f"only {differ / total:.1%} of plays changed"


@pytest.mark.parametrize("depth", [1, 2])
def test_deeper_play_still_only_returns_legal_cards(depth):
    agent = PeggingAwareAgent(play_depth=depth)
    rng = random.Random(depth)
    state = new_game(seed=depth)
    while not state.is_terminal():
        player = state.current_player
        info = state.information_state(player)
        action = agent.act(info) if player == 0 else rng.choice(info.legal)
        assert action in info.legal
        state.apply_action(action)


def test_the_controls_wreck_exactly_one_half_of_the_policy():
    """RandomPlayAgent must still discard well, and vice versa."""
    from cribbage.agents import RandomDiscardAgent, RandomPlayAgent

    baseline = HeuristicAgent()
    hand = parse_hand("5S 5H 5D 5C 2H 3C")
    info = discard_view(hand, True)

    # Random *play* leaves the discard untouched.
    assert RandomPlayAgent(seed=1).discard(info) == baseline.discard(info)
    # Random *discard* leaves the play untouched.
    state = new_game(seed=4)
    while state.phase is not Phase.PLAY:
        state.apply_action(baseline.act(
            state.information_state(state.current_player)
        ))
    play_info = state.information_state(state.current_player)
    assert RandomDiscardAgent(seed=1).play(play_info) == baseline.play(play_info)
