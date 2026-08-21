"""The playable game: input parsing, the rigged cut, and information hygiene.

An interactive feature is easy to ship untested and easy to get subtly wrong --
particularly around showing the player something they should not see -- so the
console is driven by scripted input here rather than by a terminal.
"""

from __future__ import annotations

import pytest

from cribbage.agents import make_agent
from cribbage.cards import card_str, parse_card, parse_hand
from cribbage.engine import CribbageState, Phase
from cribbage.interactive import Console, parse_selection, play_interactive
from cribbage.luck import PercentileCut


def scripted(answers=None):
    """A console that lays away the last two cards and plays the first legal one."""
    captured: list[str] = []

    def read(prompt: str) -> str:
        if answers:
            return answers.pop(0)
        return "5 6" if "lay two" in prompt else "1"

    return Console(read=read, write=captured.append), captured


# ----------------------------------------------------------------------
# Input parsing
# ----------------------------------------------------------------------


def test_parse_selection_accepts_indices_names_and_a_mixture():
    hand = parse_hand("2S 4C 5H 5D 8C TD")
    assert parse_selection("1 6", hand, 2) == [hand[0], hand[5]]
    assert parse_selection("2S,TD", hand, 2) == [parse_card("2S"), parse_card("TD")]
    assert parse_selection("1 TD", hand, 2) == [hand[0], parse_card("TD")]
    assert parse_selection("3", hand, 1) == [hand[2]]


@pytest.mark.parametrize(
    "text,wanted,message",
    [
        ("1", 2, "pick 2"),
        ("1 2 3", 2, "pick 2"),
        ("9 1", 2, "not on the list"),
        ("1 1", 2, "same card twice"),
        ("KH 1", 2, "do not hold"),
        ("banana 1", 2, "bad card"),
    ],
)
def test_parse_selection_rejects_bad_input_with_a_usable_message(text, wanted, message):
    hand = parse_hand("2S 4C 5H 5D 8C TD")
    with pytest.raises(ValueError, match=message):
        parse_selection(text, hand, wanted)


# ----------------------------------------------------------------------
# The rigged cut
# ----------------------------------------------------------------------


def test_percentile_cut_turns_the_best_and_worst_card_available():
    """At 100 it must be the single best of the forty, at 0 the single worst."""
    state = CribbageState(seed=4)
    while state.phase is Phase.DISCARD:
        state.apply_action(state.legal_actions()[0])
    context = None
    # Rebuild the pre-cut position from a fresh game, since the cut already ran.
    fresh = CribbageState(seed=4)
    while fresh.phase is Phase.DISCARD:
        if len(fresh.crib) == 4:
            break
        fresh.apply_action(fresh.legal_actions()[0])
    context = fresh.cut_context()

    from cribbage.luck import show_points

    values = {
        card: show_points(context.kept, context.crib, context.dealer, card)
        for card in context.deck
    }
    nets = {card: value[0] - value[1] for card, value in values.items()}

    assert PercentileCut(0, 100)(context) == max(nets, key=lambda c: (nets[c], c))
    assert PercentileCut(0, 0)(context) == min(nets, key=lambda c: (nets[c], c))


def test_percentile_cut_rejects_an_impossible_percentile():
    with pytest.raises(ValueError, match="percentile"):
        PercentileCut(0, 140)


@pytest.mark.parametrize("percentile,expected", [(100, 1.0), (0, -1.0)])
def test_rigging_the_cut_moves_realised_luck_the_right_way(percentile, expected):
    console, _ = scripted()
    result = play_interactive(
        make_agent("heuristic"), seed=11, percentile=percentile, console=console
    )
    assert result.finished
    assert result.luck is not None
    assert result.luck.net * expected > 0, (
        f"asked for the {percentile}th percentile, got {result.luck.net:+.1f}"
    )
    assert abs(result.luck.net_z) > 2.0, "rigging every cut should be extreme"


def test_a_fair_game_needs_no_chooser():
    console, _ = scripted()
    result = play_interactive(make_agent("heuristic"), seed=3, console=console)
    assert result.finished
    assert result.state.is_terminal()


# ----------------------------------------------------------------------
# Playing through
# ----------------------------------------------------------------------


def test_a_scripted_game_plays_to_completion():
    console, out = scripted()
    result = play_interactive(make_agent("heuristic"), seed=7, console=console)
    assert result.state.is_terminal()
    text = "\n".join(out)
    assert "wins" in text
    assert "Your cut luck" in text
    assert result.luck is not None


def test_quitting_stops_cleanly_without_a_result():
    console = Console(read=lambda prompt: "q", write=lambda line: None)
    result = play_interactive(make_agent("heuristic"), seed=7, console=console)
    assert not result.finished and result.luck is None


def test_the_display_never_reveals_the_opponents_lay_away_early():
    """The crib is face down until the show; the commentary must respect that.

    Scoped to within a round on purpose: the deck is reshuffled every deal, so a
    card laid away in round one can legitimately be dealt and played in round
    three. Only an appearance *before that round's own crib is turned up* is a
    leak.
    """
    console, out = scripted()
    result = play_interactive(
        make_agent("heuristic"), seed=5, human=0, console=console
    )

    starts = {}
    for index, line in enumerate(out):
        stripped = line.strip()
        if stripped.startswith("Round "):
            starts[int(stripped.split()[1])] = index
    assert starts, "expected round headers in the transcript"

    checked = 0
    for record in result.state.round_records:
        begin = starts.get(record.round_index)
        if begin is None:
            continue
        after = [i for r, i in starts.items() if i > begin]
        window = out[begin: min(after) if after else len(out)]

        crib_at = next((i for i, line in enumerate(window) if " crib  " in line), None)
        if crib_at is None:
            continue  # the round's show never happened
        earlier = "\n".join(window[:crib_at])

        laid = set(record.dealt[1]) - set(record.kept[1])
        assert len(laid) == 2
        for card in laid:
            assert card_str(card) not in earlier, (
                f"round {record.round_index}: {card_str(card)} was shown "
                "before the crib was turned up"
            )
        checked += 1
    assert checked >= 3, f"only checked {checked} rounds"


def test_the_deal_event_detail_is_never_printed():
    """That event carries *both* hands; printing it would hand over the game."""
    console, out = scripted()
    result = play_interactive(make_agent("heuristic"), seed=9, console=console)
    text = "\n".join(out)
    for event in result.state.events:
        if event.kind == "deal":
            assert event.detail not in text
