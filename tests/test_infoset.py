"""Information hygiene and determinization.

Two things have to hold for the engine to be usable by an imperfect-information
search:

* An information state must contain everything the player knows and nothing
  else.  The sharpest form of this is indistinguishability: two worlds that
  differ only in what the player cannot see must produce *equal* information
  states.
* A determinization must be a world the player cannot tell from the real one --
  which is the same property, stated from the other direction.
"""

from __future__ import annotations

import random

from cribbage.cards import NUM_CARDS
from cribbage.engine import Phase, new_game


def advance_to_dealer_discard(seed: int):
    """A state where the pone has discarded and the dealer is on the clock."""
    state = new_game(seed=seed)
    rng = random.Random(seed)
    assert state.current_player == state.pone
    state.apply_action(rng.choice(state.legal_actions()))
    assert state.phase is Phase.DISCARD and state.current_player == state.dealer
    return state


def dealer_view_with_pone_discard(state, pair):
    """Rebuild the position with the pone having laid away ``pair`` instead."""
    clone = state.clone()
    pone = clone.pone
    clone.hands[pone] = sorted(clone.dealt[pone])
    for card in pair:
        clone.hands[pone].remove(card)
    clone.discards[pone] = list(pair)
    clone.kept[pone] = sorted(clone.hands[pone])
    clone.crib = list(pair)
    return clone.information_state(clone.dealer)


def test_dealer_cannot_distinguish_the_pone_s_discard():
    """The whole sequential-discard model rests on this.

    Real cribbage discards simultaneously; the engine does it pone-then-dealer.
    That is only legitimate if the dealer's view is identical no matter which
    two cards the pone laid away.
    """
    from itertools import combinations

    for seed in range(20):
        state = advance_to_dealer_discard(seed)
        pone_hand = sorted(state.dealt[state.pone])
        views = {
            dealer_view_with_pone_discard(state, pair)
            for pair in combinations(pone_hand, 2)
        }
        assert len(views) == 1, (
            "the dealer's information state varies with the pone's discard"
        )


def test_information_state_never_names_a_hidden_card():
    rng = random.Random(0)
    for seed in range(20):
        state = new_game(seed=seed)
        while not state.is_terminal():
            player = state.current_player
            assert player is not None
            opp = 1 - player
            info = state.information_state(player)

            named = set(info.hand) | set(info.my_discards) | set(info.my_played)
            named |= set(info.opp_played) | set(info.seq)
            named |= {card for _, card in info.play_order}
            if info.starter is not None:
                named.add(info.starter)

            hidden = set(state.hands[opp]) | set(state.discards[opp]) | set(state.deck)
            assert named.isdisjoint(hidden), "an information state named a hidden card"
            assert set(info.unseen) >= hidden, "hidden cards must remain unseen"
            assert info.seen.isdisjoint(hidden)

            state.apply_action(rng.choice(state.legal_actions()))


def test_determinization_is_indistinguishable_from_the_real_world():
    rng = random.Random(1)
    sampler = random.Random(99)
    checked = 0
    for seed in range(15):
        state = new_game(seed=seed)
        while not state.is_terminal():
            player = state.current_player
            assert player is not None
            world = state.determinize(player, sampler)

            # The defining property: the player cannot tell the two apart.
            assert world.information_state(player) == state.information_state(player)

            # And it is a real, complete position.
            cards = (
                world.hands[0] + world.hands[1]
                + world.played[0] + world.played[1]
                + world.discards[0] + world.discards[1]
                + world.deck
                + ([world.starter] if world.starter is not None else [])
            )
            assert len(cards) == NUM_CARDS and len(set(cards)) == NUM_CARDS
            assert world.legal_actions() == state.legal_actions()
            checked += 1

            state.apply_action(rng.choice(state.legal_actions()))
    assert checked > 500, "expected to exercise many decision points"


def test_determinization_resamples_the_hidden_cards():
    """Repeated sampling should actually move the opponent's hand around."""
    state = new_game(seed=4)
    state.apply_action(state.legal_actions()[0])  # pone discards
    player = state.current_player
    sampler = random.Random(7)
    opp = 1 - player
    seen_hands = {tuple(state.determinize(player, sampler).hands[opp]) for _ in range(50)}
    assert len(seen_hands) > 1, "determinize returned the same world every time"


def test_determinized_games_can_be_played_to_completion():
    """A sampled world is a legal position, not just a consistent-looking one."""
    rng = random.Random(2)
    state = new_game(seed=8)
    for _ in range(12):
        state.apply_action(rng.choice(state.legal_actions()))
    world = state.determinize(state.current_player, random.Random(3))
    while not world.is_terminal():
        world.apply_action(rng.choice(world.legal_actions()))
    assert world.result().margin > 0


def test_clone_is_independent_and_replays_identically():
    rng = random.Random(6)
    state = new_game(seed=21)
    for _ in range(8):
        state.apply_action(rng.choice(state.legal_actions()))

    left, right = state.clone(), state.clone()
    # Same RNG state, so identical action choices give identical continuations.
    while not left.is_terminal():
        action = left.legal_actions()[0]
        left.apply_action(action)
        right.apply_action(action)
    assert left.scores == right.scores
    assert state.scores != left.scores or state.is_terminal()

    fork = state.clone()
    before = list(state.scores)
    while not fork.is_terminal():
        fork.apply_action(fork.legal_actions()[0])
    assert state.scores == before, "mutating a clone touched the original"
