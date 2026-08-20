"""Baseline agents: legality everywhere, and a few decisions with a right answer."""

from __future__ import annotations

import random
from itertools import combinations

import pytest

from cribbage.agents import AGENTS, HeuristicAgent, RandomAgent, make_agent
from cribbage.arena import play_game, run_match
from cribbage.cards import parse_card, parse_hand
from cribbage.engine import InfoState, Phase, new_game


def discard_view(hand: str, is_dealer: bool) -> InfoState:
    cards = tuple(sorted(parse_hand(hand)))
    return InfoState(
        player=0, phase=Phase.DISCARD, dealer=0 if is_dealer else 1, round_index=1,
        target=121, my_score=0, opp_score=0, hand=cards, my_discards=(), starter=None,
        count=0, seq=(), play_order=(), my_played=(), opp_played=(), opp_hand_size=6,
        legal=tuple(combinations(cards, 2)),
    )


def play_view(hand: str, count: int = 0, seq: str = "", opp_hand_size: int = 4) -> InfoState:
    cards = tuple(sorted(parse_hand(hand)))
    playable = tuple(c for c in cards if count + min((c >> 2) + 1, 10) <= 31)
    return InfoState(
        player=0, phase=Phase.PLAY, dealer=0, round_index=1, target=121,
        my_score=0, opp_score=0, hand=cards, my_discards=(),
        starter=parse_card("2D"), count=count, seq=tuple(parse_hand(seq)) if seq else (),
        play_order=(), my_played=(), opp_played=(), opp_hand_size=opp_hand_size,
        legal=playable,
    )


# ----------------------------------------------------------------------
# Legality
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(AGENTS))
def test_agents_only_ever_return_legal_actions(name):
    agent = make_agent(name, seed=1)
    opponent = RandomAgent(seed=2)
    for seed in range(4):
        state = new_game(seed=seed)
        while not state.is_terminal():
            player = state.current_player
            assert player is not None
            info = state.information_state(player)
            actor = agent if player == 0 else opponent
            action = actor.act(info)
            assert action in info.legal, f"{name} chose {action}, legal was {info.legal}"
            state.apply_action(action)


def test_agents_receive_an_information_state_not_the_engine_state():
    seen: list[object] = []

    class Spy(RandomAgent):
        def act(self, info):
            seen.append(info)
            return super().act(info)

    play_game([Spy(seed=1), RandomAgent(seed=2)], seed=5)
    assert seen and all(isinstance(info, InfoState) for info in seen)
    assert not any(hasattr(info, "hands") or hasattr(info, "crib") for info in seen)


def test_make_agent_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown agent"):
        make_agent("nonesuch")


# ----------------------------------------------------------------------
# Discard decisions with a defensible right answer
# ----------------------------------------------------------------------


def test_keeps_four_fives():
    """Four fives are worth at least 20 before the starter; nothing beats holding them."""
    agent = HeuristicAgent()
    for is_dealer in (True, False):
        laid = sorted(agent.discard(discard_view("5S 5H 5D 5C 2H 3C", is_dealer)))
        assert laid == sorted(parse_hand("2H 3C"))


def test_crib_term_is_signed_by_who_owns_the_crib():
    """The same six cards should be split differently as dealer and as pone.

    Holding A-2-3-4-5-K, the dealer lays 5-K into its *own* crib -- a five is the
    single most valuable card to put there.  The pone lays the ace instead,
    keeping 2-3-4-5, precisely to avoid handing the opponent that five.
    """
    agent = HeuristicAgent()
    as_dealer = sorted(agent.discard(discard_view("AS 2H 3D 4C 5S KH", True)))
    as_pone = sorted(agent.discard(discard_view("AS 2H 3D 4C 5S KH", False)))

    assert as_dealer == sorted(parse_hand("5S KH"))
    assert as_pone == sorted(parse_hand("AS KH"))
    assert parse_card("5S") in as_dealer and parse_card("5S") not in as_pone


def test_dealer_and_pone_disagree_often_enough_to_matter():
    """Sanity check that the sign is not a one-hand fluke."""
    agent = HeuristicAgent()
    rng = random.Random(4)
    disagreements = 0
    for _ in range(60):
        hand = rng.sample(range(52), 6)
        text = " ".join(f"{'A23456789TJQK'[c >> 2]}{'SHDC'[c & 3]}" for c in hand)
        if sorted(agent.discard(discard_view(text, True))) != sorted(
            agent.discard(discard_view(text, False))
        ):
            disagreements += 1
    assert disagreements > 10, f"only {disagreements}/60 hands split differently"


# ----------------------------------------------------------------------
# Play decisions with a defensible right answer
# ----------------------------------------------------------------------


@pytest.mark.parametrize("hand", ["5S 7H 9D KC", "5S 4H 9D KC", "5S 6H 8D QC"])
def test_does_not_lead_a_five(hand):
    """Leading a five invites any of the sixteen ten-value cards to peg fifteen.

    Nothing in the agent says "do not lead a five"; it falls out of averaging the
    opponent's reply.
    """
    agent = HeuristicAgent()
    assert agent.play(play_view(hand)) != parse_card("5S")


def test_takes_the_fifteen_when_it_is_free():
    agent = HeuristicAgent()
    chosen = agent.play(play_view("5S TH 9D KC", count=5, seq="5C"))
    assert chosen == parse_card("TH"), "should count fifteen for two"


def test_greedy_variant_ignores_the_reply():
    """`greedy` is the same agent with the risk term switched off."""
    greedy = make_agent("greedy")
    assert isinstance(greedy, HeuristicAgent) and greedy.risk_weight == 0.0
    # With no fear of the reply, leading the five is no longer discouraged.
    lead = greedy.play(play_view("5S 7H 9D KC"))
    assert lead in parse_hand("5S 7H 9D KC")


# ----------------------------------------------------------------------
# Strength
# ----------------------------------------------------------------------


def test_heuristic_beats_random_decisively():
    stats = run_match([HeuristicAgent(), RandomAgent(seed=3)], games=60, seed=1)
    low, _ = __import__("cribbage.arena", fromlist=["wilson_interval"]).wilson_interval(
        stats.wins[0], stats.games
    )
    assert low > 0.75, f"heuristic won {stats.win_rate:.0%}, CI lower bound {low:.0%}"


def test_heuristic_hand_scores_land_near_the_known_average():
    """A well-played cribbage hand averages roughly eight points at the show.

    This is an end-to-end check against outside knowledge: a rules bug the unit
    tests missed would almost certainly push this number off.
    """
    stats = run_match([HeuristicAgent(), HeuristicAgent()], games=40, seed=2)
    for seat in (0, 1):
        per_round = stats.points_per_round(seat, "hand")
        assert 7.0 < per_round < 9.0, f"seat {seat} averaged {per_round:.2f} per hand"
