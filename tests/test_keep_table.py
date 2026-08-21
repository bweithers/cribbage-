"""The keep-quality table and the opponent model built on it."""

from __future__ import annotations

import statistics
from itertools import combinations

import pytest

from cribbage.agents import PimcPeggingAgent
from cribbage.cards import parse_hand, rank_of
from cribbage.engine import InfoState, Phase
from cribbage.keep_table import keep_value, load_keep_table


def ranks(text: str) -> list[int]:
    return sorted(rank_of(card) for card in parse_hand(text))


def test_every_four_card_keep_is_covered():
    assert len(load_keep_table()) == 1820


def test_it_ranks_keeps_the_way_a_player_would():
    assert keep_value(ranks("5S 5H 5D JC")) > keep_value(ranks("AS 2H 3D 4C"))
    assert keep_value(ranks("AS 2H 3D 4C")) > keep_value(ranks("2S 7H 9D KC"))
    # The worst keeps are disconnected and ten-heavy, but never worth nothing:
    # 6-T-T-K still averages over three, because the pair alone is two.
    table = load_keep_table()
    worst = min(table, key=table.get)
    assert 3.0 < table[worst] < 4.0, f"worst keep scored {table[worst]:.2f}"
    assert keep_value(ranks("6S TH TD KC")) == pytest.approx(table[worst], abs=0.01)
    assert table[max(table, key=table.get)] > 22, "four fives is the best keep"


def play_view(hand, opp_played=(), opp_hand_size=4) -> InfoState:
    cards = tuple(sorted(hand))
    return InfoState(
        player=0, phase=Phase.PLAY, dealer=0, round_index=1, target=121,
        my_score=0, opp_score=0, hand=cards, my_discards=(),
        starter=parse_hand("2D")[0], count=0, seq=(), play_order=(),
        my_played=(), opp_played=tuple(opp_played),
        opp_hand_size=opp_hand_size, legal=cards,
    )


def test_uniform_sampling_draws_the_right_shape():
    agent = PimcPeggingAgent(samples=8, opponent_beta=0.0, seed=1)
    info = play_view(parse_hand("5S 6H 7D 8C"))
    unseen = list(info.unseen)
    drawn = agent.sample_holdings(info, unseen, 4, 10)
    assert len(drawn) == 10
    for holding in drawn:
        assert len(set(holding)) == 4
        assert set(holding) <= set(unseen)


def test_weighting_biases_the_guess_toward_hands_worth_keeping():
    """The whole point of the opponent model: they chose four cards from six."""
    info = play_view(parse_hand("5S 6H 7D 8C"))
    unseen = list(info.unseen)

    def mean_quality(beta: float) -> float:
        agent = PimcPeggingAgent(samples=8, opponent_beta=beta, seed=7)
        drawn = agent.sample_holdings(info, unseen, 4, 300)
        return statistics.mean(
            keep_value(sorted(rank_of(card) for card in holding))
            for holding in drawn
        )

    uniform = mean_quality(0.0)
    weighted = mean_quality(1.0)
    assert weighted > uniform + 0.5, (
        f"weighted draws averaged {weighted:.2f} against uniform {uniform:.2f}"
    )


def test_an_empty_opponent_hand_needs_no_guessing():
    agent = PimcPeggingAgent(samples=24, opponent_beta=1.0, seed=1)
    info = play_view(parse_hand("5S 6H"), opp_hand_size=0)
    assert agent.sample_holdings(info, list(info.unseen), 0, 24) == [[]]


def test_the_model_accounts_for_cards_already_played():
    """A keep is four cards; the ones already on the table are part of it."""
    played = parse_hand("KS")
    info = play_view(parse_hand("5S 6H 7D"), opp_played=played, opp_hand_size=3)
    agent = PimcPeggingAgent(samples=4, opponent_beta=1.0, seed=2)
    drawn = agent.sample_holdings(info, list(info.unseen), 3, 12)
    assert all(len(holding) == 3 for holding in drawn)
    assert all(set(holding).isdisjoint(played) for holding in drawn)


@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_the_agent_still_plays_legally_under_either_model(beta):
    import random

    from cribbage.engine import new_game

    agent = PimcPeggingAgent(samples=4, opponent_beta=beta, seed=3)
    rng = random.Random(1)
    state = new_game(seed=11)
    while not state.is_terminal():
        player = state.current_player
        info = state.information_state(player)
        action = agent.act(info) if player == 0 else rng.choice(info.legal)
        assert action in info.legal
        state.apply_action(action)


def test_discard_legality_is_untouched_by_the_opponent_model():
    cards = tuple(sorted(parse_hand("5S 5H 5D 5C 2H 3C")))
    info = InfoState(
        player=0, phase=Phase.DISCARD, dealer=0, round_index=1, target=121,
        my_score=0, opp_score=0, hand=cards, my_discards=(), starter=None,
        count=0, seq=(), play_order=(), my_played=(), opp_played=(),
        opp_hand_size=6, legal=tuple(combinations(cards, 2)),
    )
    assert PimcPeggingAgent(opponent_beta=1.0).discard(info) in info.legal
