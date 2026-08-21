"""A playable game in the terminal, with the cut rigged to a chosen quality.

The point is to feel what a given level of cut luck is actually like, so the
cut is not left to a seed.  A seed has to be fixed before anyone discards and
therefore cannot know what a card will be worth; the cut happens *after* both
discards, when the holdings and crib are settled and all forty remaining cards
can be scored exactly.  ``PercentileCut`` turns the one at the percentile you
asked for, every round.

Reading and writing go through injected callables so the whole loop can be
driven by a test with scripted input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from .agents.base import Agent
from .cards import card_str, parse_card
from .engine import DEFAULT_TARGET, CribbageState, Event, InfoState, Phase
from .luck import CutLuck, PercentileCut, cut_luck

__all__ = ["Console", "InteractiveResult", "play_interactive", "parse_selection"]

RULE = "─" * 62


@dataclass
class InteractiveResult:
    """What a finished session produced.  ``luck`` is None if the player quit."""

    state: CribbageState
    luck: Optional[CutLuck]

    @property
    def finished(self) -> bool:
        return self.luck is not None


class QuitGame(Exception):
    """Raised when the player asks to stop, or input runs out."""


@dataclass
class Console:
    """Terminal I/O, injectable so tests can script a whole game."""

    read: Callable[[str], str] = input
    write: Callable[[str], None] = print

    def ask(self, prompt: str) -> str:
        try:
            answer = self.read(prompt)
        except (EOFError, KeyboardInterrupt):
            raise QuitGame from None
        if answer.strip().lower() in ("q", "quit", "exit"):
            raise QuitGame
        return answer

    def say(self, text: str = "") -> None:
        self.write(text)


def parse_selection(text: str, displayed: Sequence[int], wanted: int) -> list[int]:
    """Read ``wanted`` cards from ``text``, by 1-based index or by name.

    Accepts ``"1 4"``, ``"1,4"``, ``"5H JS"`` and mixtures.  Raises ``ValueError``
    with something a human can act on.
    """
    tokens = text.replace(",", " ").split()
    if len(tokens) != wanted:
        raise ValueError(
            f"pick {wanted} card{'s' if wanted > 1 else ''}, "
            f"you gave {len(tokens)}"
        )

    chosen: list[int] = []
    for token in tokens:
        if token.isdigit():
            index = int(token)
            if not 1 <= index <= len(displayed):
                raise ValueError(f"{index} is not on the list")
            chosen.append(displayed[index - 1])
        else:
            card = parse_card(token)  # raises ValueError on nonsense
            if card not in displayed:
                raise ValueError(f"you do not hold {card_str(card)}")
            chosen.append(card)

    if len(set(chosen)) != wanted:
        raise ValueError("that is the same card twice")
    return chosen


def _hand_menu(cards: Sequence[int], playable: Optional[Sequence[int]] = None) -> str:
    parts = []
    for index, card in enumerate(cards, start=1):
        label = f"[{index}] {card_str(card)}"
        if playable is not None and card not in playable:
            label = f" {'-' * (len(label) - 1)}"  # too big for the count
        parts.append(f"{label:<9}")
    return "".join(parts).rstrip()


class _Renderer:
    """Turns the engine's event log into a running commentary.

    Careful about what it is allowed to show: the deal event carries both hands
    and the opponent's discard carries their lay-away, neither of which the
    player may see.  Everything from the cut onward is public at the table.
    """

    def __init__(self, console: Console, human: int, names: Sequence[str]):
        self.console = console
        self.human = human
        self.names = names
        self.position = 0
        self.scores = [0, 0]

    def who(self, player: int) -> str:
        return self.names[player]

    def verb(self, player: int, base: str) -> str:
        """"you play" but "heuristic plays"."""
        return base if player == self.human else base + "s"

    def possessive(self, player: int) -> str:
        return "yours" if player == self.human else f"{self.names[player]}'s"

    def flush(self, state: CribbageState) -> None:
        for event in state.events[self.position:]:
            self._render(event, state)
        self.position = len(state.events)

    def _score_tag(self, event: Event) -> str:
        if not event.points:
            return ""
        self.scores[event.player] += event.points
        return (
            f"   +{event.points}   "
            f"({self.names[0]} {self.scores[0]} · {self.names[1]} {self.scores[1]})"
        )

    def _render(self, event: Event, state: CribbageState) -> None:
        say = self.console.say
        kind = event.kind

        if kind == "deal":
            say()
            say(f"{RULE}")
            say(
                f"  Round {event.round}     "
                f"{self.who(event.player)} {self.verb(event.player, 'deal')}, "
                f"so the crib is {self.possessive(event.player)}"
            )
            return
        if kind == "discard":
            if event.player == self.human:
                say(f"  you lay away  {event.detail}")
            else:
                say(f"  {self.who(event.player)} "
                    f"{self.verb(event.player, 'lay')} two away")
            return
        if kind == "cut":
            say(f"  cut  {event.detail}")
            return
        if kind == "heels":
            say(f"  {self.who(event.player)} — his heels{self._score_tag(event)}")
            return
        if kind == "play":
            say(f"  {self.who(event.player)} {self.verb(event.player, 'play')} "
                f"{event.detail}{self._score_tag(event)}")
            return
        if kind == "reset":
            say("        — count resets —")
            return
        if kind == "go":
            say(f"  {self.who(event.player)} — {event.detail}{self._score_tag(event)}")
            return
        if kind in ("hand", "crib"):
            label = "crib" if kind == "crib" else "hand"
            say(f"  {self.who(event.player)} {label}  {event.detail}"
                f"{self._score_tag(event)}")
            return
        if kind == "win":
            say()
            say(f"  {self.who(event.player)} wins.")


def _ask_discard(console: Console, info: InfoState) -> tuple[int, ...]:
    whose = "yours" if info.is_dealer else "theirs"
    console.say()
    console.say(f"  your hand   {_hand_menu(info.hand)}")
    while True:
        answer = console.ask(f"  lay two away (crib is {whose}) › ")
        try:
            chosen = parse_selection(answer, info.hand, 2)
        except ValueError as problem:
            console.say(f"    {problem}. Try '1 4' or '5H JS'.")
            continue
        pair = tuple(sorted(chosen))
        if pair not in info.legal:
            console.say("    that is not a legal lay-away.")
            continue
        return pair


def _ask_play(console: Console, info: InfoState) -> int:
    while True:
        console.say(
            f"  count {info.count:<3}  {_hand_menu(info.hand, info.legal)}"
        )
        answer = console.ask("  play › ")
        try:
            chosen = parse_selection(answer, info.hand, 1)
        except ValueError as problem:
            console.say(f"    {problem}. Try '1' or '5H'.")
            continue
        card = chosen[0]
        if card not in info.legal:
            console.say(
                f"    {card_str(card)} would take the count past 31."
            )
            continue
        return card


def play_interactive(
    opponent: Agent,
    *,
    seed: Optional[int] = None,
    human: int = 0,
    dealer: int = 0,
    target: int = DEFAULT_TARGET,
    percentile: Optional[float] = None,
    console: Optional[Console] = None,
) -> InteractiveResult:
    """Play a whole game against ``opponent``.  Returns the state and cut luck.

    ``percentile`` rigs every cut to that quality for the human; leave it out
    for an honest game.
    """
    console = console or Console()
    names = ["you", opponent.name] if human == 0 else [opponent.name, "you"]

    chooser = None if percentile is None else PercentileCut(human, percentile)
    state = CribbageState(
        dealer=dealer, target=target, seed=seed, cut_chooser=chooser
    )
    renderer = _Renderer(console, human, names)

    console.say()
    console.say(f"  Cribbage to {target}. You are {names[human]}; "
                f"'q' quits at any prompt.")
    if percentile is not None:
        console.say(
            f"  Every cut is rigged to the {percentile:g}th percentile of what it "
            f"could have been for you."
        )

    try:
        while not state.is_terminal():
            renderer.flush(state)
            player = state.current_player
            assert player is not None
            info = state.information_state(player)
            if player == human:
                action = (
                    _ask_discard(console, info)
                    if info.phase is Phase.DISCARD
                    else _ask_play(console, info)
                )
            else:
                action = opponent.act(info)
            state.apply_action(action)
        renderer.flush(state)
    except QuitGame:
        console.say("\n  Stopped.")
        return InteractiveResult(state=state, luck=None)

    return InteractiveResult(
        state=state,
        luck=_summarise(console, state, human, names, percentile),
    )


def _summarise(
    console: Console,
    state: CribbageState,
    human: int,
    names: Sequence[str],
    percentile: Optional[float],
) -> CutLuck:
    result = state.result()
    luck = cut_luck(state, human)

    console.say()
    console.say(RULE)
    verdict = "you win" if result.winner == human else f"{names[1 - human]} wins"
    tail = "  (double skunk!)" if result.double_skunk else (
        "  (skunk)" if result.skunk else ""
    )
    console.say(
        f"  {names[0]} {result.scores[0]} · {names[1]} {result.scores[1]}"
        f"   — {verdict}{tail}"
    )
    console.say()
    console.say(f"  Your cut luck: {luck.describe()}")

    if percentile is not None and luck.rounds:
        console.say(
            f"\n  You asked for every cut at the {percentile:g}th percentile. Over "
            f"{len(luck.rounds)} rounds that\n  compounds: per-round edges add "
            f"while their spreads only add in quadrature,\n  so a steady "
            f"{percentile:g}th-percentile cut lands near z = {luck.net_z:+.1f} "
            f"for the game."
        )
    return luck
