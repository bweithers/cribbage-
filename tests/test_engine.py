"""The engine: scripted play-phase scenarios, invariants, and information hygiene.

The scenarios drive exact hands rather than random ones, so every branch of the
play loop -- go, thirty-one, last card, reset -- is reached deliberately instead
of by luck.
"""

from __future__ import annotations

import random

import pytest

from cribbage.cards import NUM_CARDS, parse_card, parse_hand
from cribbage.engine import (
    MAX_COUNT,
    CribbageState,
    Phase,
    new_game,
)
from cribbage.scoring import score_play


def drive(state: CribbageState, cards: str) -> CribbageState:
    """Play the named cards in order, checking each is legal and count stays sane."""
    for card in parse_hand(cards):
        assert state.phase is Phase.PLAY, "play ended earlier than the script expected"
        legal = state.legal_actions()
        assert card in legal, (
            f"{card} is not legal; legal actions are {legal} at count {state.count}"
        )
        state.apply_action(card)
        assert state.count <= MAX_COUNT
    return state


def play_events(state: CribbageState) -> list[tuple[int, int, str]]:
    """(player, points, kind) for the pegging events only, ignoring the show."""
    return [
        (e.player, e.points, e.kind)
        for e in state.events
        if e.kind in ("play", "go")
    ]


def play_points(state: CribbageState) -> list[int]:
    totals = [0, 0]
    for player, points, _ in play_events(state):
        totals[player] += points
    return totals


def assert_cards_conserved(state: CribbageState) -> None:
    """Every one of the 52 cards is in exactly one place."""
    seen: list[int] = []
    seen += state.hands[0] + state.hands[1]
    seen += state.played[0] + state.played[1]
    seen += state.discards[0] + state.discards[1]
    seen += state.deck
    if state.starter is not None:
        seen.append(state.starter)
    assert len(seen) == NUM_CARDS, f"{len(seen)} cards accounted for, expected 52"
    assert len(set(seen)) == NUM_CARDS, "a card appears in two places at once"


# ----------------------------------------------------------------------
# Scripted play-phase scenarios
# ----------------------------------------------------------------------


def test_a_go_is_not_a_turn_change():
    """The player who can still play keeps playing, and scores off their own run.

    Pone holds nothing but ten-value cards, so once the count passes 21 the
    dealer lays 2-3-4 on three consecutive turns and pegs a run of three that
    only exists because the pone kept saying go.
    """
    state = CribbageState.for_play(
        hands=[parse_hand("AS 2S 3S 4S"), parse_hand("KS KH KD QS")],
        starter=parse_card("9H"),
        dealer=0,
    )
    drive(state, "KS AS KH 2S 3S 4S KD QS")

    assert play_events(state) == [
        (1, 0, "play"),   # KS -> 10
        (0, 0, "play"),   # AS -> 11
        (1, 0, "play"),   # KH -> 21
        (0, 0, "play"),   # 2S -> 23, pone can no longer play
        (0, 0, "play"),   # 3S -> 26, dealer plays again
        (0, 3, "play"),   # 4S -> 30, run of three off the dealer's own cards
        (0, 1, "go"),     # neither can play
        (1, 0, "play"),   # KD -> 10, pone leads the new sub-round
        (1, 0, "play"),   # QS -> 20, dealer is out of cards
        (1, 1, "go"),     # last card
    ]
    assert play_points(state) == [4, 1]


def test_thirty_one_pays_two_and_not_also_a_go_point():
    """Hitting 31 scores 2, resets, and hands the lead to the other player."""
    state = CribbageState.for_play(
        hands=[parse_hand("KS KH QD QC"), parse_hand("5S 6S 7S 8S")],
        starter=parse_card("2H"),
        dealer=0,
    )
    drive(state, "5S KS 6S KH 7S QD 8S QC")

    assert play_events(state) == [
        (1, 0, "play"),   # 5S  -> 5
        (0, 2, "play"),   # KS  -> 15, two for fifteen
        (1, 0, "play"),   # 6S  -> 21
        (0, 2, "play"),   # KH  -> 31, two and only two
        (1, 0, "play"),   # 7S  -> 7, pone leads after the thirty-one
        (0, 0, "play"),   # QD  -> 17
        (1, 0, "play"),   # 8S  -> 25
        (1, 1, "go"),     # dealer's queen will not fit
        (0, 0, "play"),   # QC  -> 10
        (0, 1, "go"),     # last card
    ]
    assert play_points(state) == [5, 1]


def test_lead_after_thirty_one_goes_to_the_other_player():
    state = CribbageState.for_play(
        hands=[parse_hand("KS KH QD QC"), parse_hand("5S 6S 7S 8S")],
        starter=parse_card("2H"),
        dealer=0,
    )
    drive(state, "5S KS 6S KH")
    assert state.count == 0, "the count resets immediately on thirty-one"
    assert state.seq == []
    assert state.current_player == 1, "dealer played the 31, so pone leads next"


def test_a_final_card_that_makes_thirty_one_pays_no_last_card_point():
    """All eight cards land in one sub-round ending exactly on 31."""
    state = CribbageState.for_play(
        hands=[parse_hand("8S 8H 8D 3S"), parse_hand("AS AH AD AC")],
        starter=parse_card("KH"),
        dealer=0,
    )
    drive(state, "AS 8S AH 8H AD 8D AC 3S")

    kinds = [kind for _, _, kind in play_events(state)]
    assert "go" not in kinds, "no go or last-card point when the play ends on 31"
    assert play_points(state) == [2, 0], "the dealer's two for thirty-one, nothing else"


def test_pairs_do_not_reach_across_a_reset():
    """A card laid after a reset cannot pair with one from the previous sub-round."""
    state = CribbageState.for_play(
        hands=[parse_hand("KD KC QS QH"), parse_hand("7S 7H KS KH")],
        starter=parse_card("2D"),
        dealer=0,
    )
    drive(state, "KS KD KH KC 7S QS 7H QH")

    assert play_events(state) == [
        (1, 0, "play"),   # KS -> 10
        (0, 2, "play"),   # KD -> 20, pair
        (1, 6, "play"),   # KH -> 30, three of a kind
        (1, 1, "go"),     # nothing else fits
        (0, 0, "play"),   # KC -> 10, the fourth king pairs with nothing: reset wall
        (1, 0, "play"),   # 7S -> 17
        (0, 0, "play"),   # QS -> 27
        (0, 1, "go"),
        (1, 0, "play"),   # 7H -> 7, does not pair with the 7S behind the wall
        (0, 0, "play"),   # QH -> 17
        (0, 1, "go"),     # last card
    ]
    assert play_points(state) == [4, 7]


def test_current_player_always_has_a_legal_action():
    """A player who cannot play is skipped, never asked."""
    rng = random.Random(11)
    for seed in range(30):
        state = new_game(seed=seed)
        while not state.is_terminal():
            legal = state.legal_actions()
            assert legal, "current_player was offered no legal action"
            assert state.current_player is not None
            state.apply_action(rng.choice(legal))


def test_play_points_match_an_independent_recomputation():
    """Replay each round's play order and check the engine's arithmetic."""
    rng = random.Random(5)
    for seed in range(20):
        state = new_game(seed=seed)
        while not state.is_terminal():
            state.apply_action(rng.choice(state.legal_actions()))

        # Group play events by round and rebuild the sub-round sequences.
        by_round: dict[int, list] = {}
        for event in state.events:
            if event.kind in ("play", "reset"):
                by_round.setdefault(event.round, []).append(event)

        for events in by_round.values():
            seq: list[int] = []
            for event in events:
                if event.kind == "reset":
                    seq = []
                    continue
                card = parse_card(event.detail.split()[0])
                seq.append(card)
                assert event.points == score_play(seq), (
                    f"engine paid {event.points} for {event.detail}, "
                    f"recomputation says {score_play(seq)}"
                )


# ----------------------------------------------------------------------
# Whole-game invariants
# ----------------------------------------------------------------------


def test_random_games_hold_their_invariants():
    rng = random.Random(3)
    for seed in range(60):
        state = new_game(seed=seed)
        previous = [0, 0]
        while not state.is_terminal():
            assert_cards_conserved(state)
            assert state.scores[0] >= previous[0] and state.scores[1] >= previous[1], (
                "scores must never go down"
            )
            previous = list(state.scores)
            state.apply_action(rng.choice(state.legal_actions()))

        result = state.result()
        assert state.scores[result.winner] >= state.target
        assert state.scores[result.loser] < state.target, (
            "the game must stop the instant someone crosses the target"
        )
        assert result.margin > 0
        assert state.returns() == ((1.0, -1.0) if result.winner == 0 else (-1.0, 1.0))


def test_every_card_is_played_each_round():
    rng = random.Random(7)
    state = new_game(seed=42)
    while not state.is_terminal():
        state.apply_action(rng.choice(state.legal_actions()))
    counts: dict[int, int] = {}
    for event in state.events:
        if event.kind == "play":
            counts[event.round] = counts.get(event.round, 0) + 1
    # Every round that reached the play phase and finished it laid all 8 cards.
    finished = [r for r in counts if r < state.round_index]
    assert finished, "expected at least one completed round"
    for r in finished:
        assert counts[r] == 8, f"round {r} played {counts[r]} cards, expected 8"


def test_dealer_alternates_between_rounds():
    rng = random.Random(9)
    state = new_game(seed=13, dealer=0)
    seen: list[tuple[int, int]] = []
    while not state.is_terminal():
        if not seen or seen[-1][0] != state.round_index:
            seen.append((state.round_index, state.dealer))
        state.apply_action(rng.choice(state.legal_actions()))
    for (r1, d1), (r2, d2) in zip(seen, seen[1:]):
        assert r2 == r1 + 1 and d2 == 1 - d1, "the deal must alternate every round"


def test_illegal_actions_are_rejected():
    state = new_game(seed=1)
    with pytest.raises(ValueError):
        state.apply_action((99, 100))
    with pytest.raises(ValueError):
        state.apply_action(("not", "cards"))

    state = CribbageState.for_play(
        hands=[parse_hand("AS KS KH KD"), parse_hand("9S 9H 9D 9C")],
        starter=parse_card("2H"),
        dealer=0,
    )
    with pytest.raises(ValueError):
        state.apply_action(parse_card("KS"))  # not in the pone's hand

    drive(state, "9S KS 9H")
    # At 28 the dealer holds one card that fits and three that do not.
    assert state.count == 28
    assert state.legal_actions() == [parse_card("AS")]
    with pytest.raises(ValueError):
        state.apply_action(parse_card("KH"))  # 28 + 10 would bust 31


# ----------------------------------------------------------------------
# Independent deal and cut streams
# ----------------------------------------------------------------------


def deal_and_starter(state: CribbageState):
    """Advance past both discards so the starter is turned, and report both."""
    hands = [sorted(state.hands[0]), sorted(state.hands[1])]
    state.apply_action(state.legal_actions()[0])
    state.apply_action(state.legal_actions()[0])
    return hands, state.starter


def test_cut_seed_varies_the_starter_without_touching_the_deal():
    """The premise of the luck decomposition in scripts/experiment_luck.py.

    Holding the deal and re-randomizing only the cut has to mean exactly that,
    or the two factors cannot be told apart.
    """
    left_hands, left_starter = deal_and_starter(new_game(seed=5, cut_seed=1))
    right_hands, right_starter = deal_and_starter(new_game(seed=5, cut_seed=2))
    assert left_hands == right_hands, "the deal must not depend on the cut stream"
    assert left_starter != right_starter


def test_deal_seed_varies_the_deal_without_touching_the_cut_stream():
    """The cut stream draws by index, and the deck always holds 40 cards at the
    cut, so a fixed cut seed picks the same *position* whatever was dealt."""
    seen = set()
    for deal_seed in range(6):
        hands, _ = deal_and_starter(new_game(seed=deal_seed, cut_seed=77))
        seen.add(tuple(hands[0]))
    assert len(seen) == 6, "each deal seed should produce a different hand"


def test_a_single_seed_still_drives_the_whole_game():
    """Without cut_seed the two streams stay shared, so one seed reproduces
    everything -- the behaviour every other test and script relies on."""
    first = deal_and_starter(new_game(seed=11))
    second = deal_and_starter(new_game(seed=11))
    assert first == second


def test_clone_preserves_split_streams():
    state = new_game(seed=5, cut_seed=3)
    rng = random.Random(0)
    for _ in range(6):
        state.apply_action(rng.choice(state.legal_actions()))

    left, right = state.clone(), state.clone()
    while not left.is_terminal():
        action = left.legal_actions()[0]
        left.apply_action(action)
        right.apply_action(action)
    assert left.scores == right.scores
    assert left.starter == right.starter


def test_cut_indices_prescribe_one_round_at_a_time():
    """Finer-grained than cut_seed, which fixes the whole sequence at once.

    Changing the index for round three must change that starter and leave the
    earlier ones alone, which is what lets an analysis separate the effect of a
    single cut from the effect of every cut in the game.
    """
    def starters(indices, rounds=3):
        state = new_game(seed=5, cut_indices=indices)
        seen = []
        for _ in range(rounds):
            while state.phase is Phase.DISCARD:
                state.apply_action(state.legal_actions()[0])
            seen.append(state.starter)
            current = state.round_index
            while state.round_index == current and not state.is_terminal():
                state.apply_action(state.legal_actions()[0])
        return seen

    base = starters([7] * 12)
    changed = starters([7, 7, 19] + [7] * 9)
    assert base[:2] == changed[:2], "earlier rounds must be untouched"
    assert base[2] != changed[2], "round three's starter must change"


def test_cut_indices_wrap_modulo_the_deck():
    """The deck always holds 40 cards at the cut, so 43 and 3 are the same cut."""
    def first_starter(indices):
        state = new_game(seed=5, cut_indices=indices, cut_seed=1)
        state.apply_action(state.legal_actions()[0])
        state.apply_action(state.legal_actions()[0])
        return state.starter

    assert first_starter([3]) == first_starter([43]), "43 mod 40 is 3"
    assert first_starter([3]) != first_starter([4])


def test_cut_indices_fall_back_to_the_stream_once_exhausted():
    """A short list must not fail, and must not steal draws from the deal."""
    prescribed = new_game(seed=5, cut_indices=[3], cut_seed=1)
    streamed = new_game(seed=5, cut_seed=1)
    rng = random.Random(0)
    for state in (prescribed, streamed):
        while state.round_index < 3 and not state.is_terminal():
            state.apply_action(rng.choice(state.legal_actions()))
    # Round one was prescribed, so the deals still match afterwards: the
    # prescribed cut consumed nothing from the deal stream.
    assert prescribed.dealt == streamed.dealt
