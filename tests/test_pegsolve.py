"""The exact pegging solver, checked against the real engine.

``cribbage/pegsolve.py`` is a second implementation of the play rules, which is
exactly the duplication this project otherwise avoids. It earns its place by
being far too hot a path to drive the real engine through, and it is held honest
here: both sides play the solver's own chosen line through the real
:class:`~cribbage.engine.CribbageState`, and the points that actually come out
must equal the value the solver predicted before a card was laid.

That is a sharp test. A disagreement about who takes the go, when the count
resets, whether 31 also pays a last-card point, or whether a run may reach across
a reset would all break the equality.
"""

from __future__ import annotations

import random

import pytest

from cribbage.cards import CARD_VALUES, parse_hand
from cribbage.engine import CribbageState, Phase
from cribbage.pegsolve import best_cards, solve


def pegging_net(state: CribbageState, seat: int) -> int:
    """Points seat scored in the play, minus the opponent's."""
    pegged = [0, 0]
    for event in state.events:
        if event.round == 1 and event.kind in ("play", "go"):
            pegged[event.player] += event.points
    return pegged[seat] - pegged[1 - seat]


def play_out_optimally(hands, starter, dealer):
    """Drive the real engine, both sides consulting the solver at every turn."""
    state = CribbageState.for_play(hands=hands, starter=starter, dealer=dealer)
    while state.phase is Phase.PLAY:
        mover = state.current_player
        assert mover is not None
        last = None if state.last_to_play is None else (
            0 if state.last_to_play == mover else 1
        )
        values = best_cards(
            state.hands[mover], state.hands[1 - mover],
            count=state.count, seq=state.seq, last=last,
        )
        assert values, "the engine offered a turn with no legal card"
        state.apply_action(max(values, key=lambda card: (values[card], -card)))
    return state


@pytest.mark.parametrize("seed", range(25))
def test_the_solvers_value_is_what_the_engine_actually_pays(seed):
    rng = random.Random(seed)
    deck = list(range(52))
    rng.shuffle(deck)
    hands = [sorted(deck[:4]), sorted(deck[4:8])]
    starter = deck[8]
    dealer = seed % 2
    pone = 1 - dealer

    predicted = solve(hands[0], hands[1], my_turn=(pone == 0))
    state = play_out_optimally(hands, starter, dealer)
    assert pegging_net(state, 0) == predicted


def test_it_holds_for_awkward_hands_too():
    """Hands built to force gos, resets and a thirty-one."""
    cases = [
        (["KS KH KD QS", "AS AH AD AC"], "2D", 0),
        (["8S 8H 8D 3S", "AS AH AD AC"], "KH", 0),   # all eight land on 31
        (["7S 7H 7D 7C", "8S 8H 8D 8C"], "2D", 1),   # four of a kind both ways
        (["AS 2S 3S 4S", "KS KH KD KC"], "9H", 0),   # repeated gos
        (["5S 5H 5D 5C", "TS TH TD TC"], "3D", 1),   # every card makes fifteen
    ]
    for hands_text, starter_text, dealer in cases:
        hands = [parse_hand(text) for text in hands_text]
        starter = parse_hand(starter_text)[0]
        pone = 1 - dealer
        predicted = solve(hands[0], hands[1], my_turn=(pone == 0))
        state = play_out_optimally(hands, starter, dealer)
        assert pegging_net(state, 0) == predicted, hands_text


def test_the_solver_is_zero_sum():
    """Whatever the value is to me, it is exactly its negative to them."""
    rng = random.Random(99)
    for _ in range(30):
        deck = list(range(52))
        rng.shuffle(deck)
        mine, theirs = sorted(deck[:4]), sorted(deck[4:8])
        for my_turn in (True, False):
            ours = solve(mine, theirs, my_turn=my_turn)
            # From the other seat, the same position with the roles exchanged.
            theirs_value = solve(theirs, mine, my_turn=not my_turn)
            assert ours == -theirs_value


def test_only_legal_cards_are_ever_valued():
    hand = parse_hand("KS KH QD 3C")
    values = best_cards(hand, parse_hand("AS 2H 4D 5C"), count=25)
    assert set(values) == {card for card in hand if CARD_VALUES[card] <= 6}


def test_perfect_play_beats_greedy_play_in_the_same_position():
    """Sanity: the solver's line is at least as good as taking points now."""
    from cribbage.scoring import score_play

    rng = random.Random(7)
    solver_better = 0
    for _ in range(60):
        deck = list(range(52))
        rng.shuffle(deck)
        mine, theirs = sorted(deck[:4]), sorted(deck[4:8])
        values = best_cards(mine, theirs)
        greedy = max(mine, key=lambda card: (score_play((card,)), -card))
        if values and greedy in values:
            if values[max(values, key=values.get)] > values[greedy]:
                solver_better += 1
    assert solver_better > 0, "the solver should sometimes decline free points"


# ----------------------------------------------------------------------
# The agent built on the solver
# ----------------------------------------------------------------------


def test_pimc_plays_perfectly_once_the_opponent_is_out_of_cards():
    """With nothing hidden left, sampling is pointless and the solve is exact.

    So the agent must find the same card the solver does, and must not waste
    repeated draws finding it.
    """
    from cribbage.agents import PimcPeggingAgent
    from cribbage.engine import InfoState, Phase

    hand = tuple(sorted(parse_hand("5S 6H 7D")))
    info = InfoState(
        player=0, phase=Phase.PLAY, dealer=0, round_index=1, target=121,
        my_score=0, opp_score=0, hand=hand, my_discards=(),
        starter=parse_hand("2D")[0], count=4, seq=tuple(parse_hand("4C")),
        play_order=((1, parse_hand("4C")[0]),), my_played=(),
        opp_played=tuple(parse_hand("4C")), opp_hand_size=0,
        legal=hand,
    )
    agent = PimcPeggingAgent(samples=24)
    values = best_cards(hand, [], count=4, seq=parse_hand("4C"), last=1)
    assert agent.play(info) == max(values, key=lambda c: (values[c], -c))


@pytest.mark.parametrize("seed", [0, 1])
def test_pimc_only_returns_legal_cards(seed):
    from cribbage.agents import PimcPeggingAgent
    from cribbage.engine import new_game

    agent = PimcPeggingAgent(samples=6, seed=seed)
    rng = random.Random(seed)
    state = new_game(seed=seed)
    while not state.is_terminal():
        player = state.current_player
        info = state.information_state(player)
        action = agent.act(info) if player == 0 else rng.choice(info.legal)
        assert action in info.legal
        state.apply_action(action)


def test_pimc_inherits_the_pegging_aware_discard():
    """Its discard must match PeggingAwareAgent's, so a match between them
    measures the search and nothing else."""
    from itertools import combinations

    from cribbage.agents import PeggingAwareAgent, PimcPeggingAgent
    from cribbage.engine import InfoState, Phase

    rng = random.Random(5)
    for _ in range(25):
        cards = tuple(sorted(rng.sample(range(52), 6)))
        info = InfoState(
            player=0, phase=Phase.DISCARD, dealer=0, round_index=1, target=121,
            my_score=0, opp_score=0, hand=cards, my_discards=(), starter=None,
            count=0, seq=(), play_order=(), my_played=(), opp_played=(),
            opp_hand_size=6, legal=tuple(combinations(cards, 2)),
        )
        assert (
            PimcPeggingAgent(peg_weight=0.5).discard(info)
            == PeggingAwareAgent(peg_weight=0.5).discard(info)
        )
