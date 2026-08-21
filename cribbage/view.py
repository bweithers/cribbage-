"""What one player is allowed to see, in a form any front end can render.

The engine's event log is a god's-eye record: the deal event carries *both*
hands and a discard event names the cards that went face down into the crib.
Handing that to a player would give away the game, so something has to redact
it -- and that something should exist once, not once per interface.

This module is that place.  :func:`visible_events` produces the redacted log and
:func:`game_view` produces a JSON-ready snapshot; the terminal client and the
web client both consume them and differ only in how they draw the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .cards import card_str, hand_str, rank_of, suit_of
from .engine import CribbageState, Phase
from .luck import CutLuck, cut_luck

__all__ = ["VisibleEvent", "visible_events", "game_view", "card_json"]

#: Events whose detail is safe to show anyone.  Everything from the cut onward
#: is face up at a real table, and the show reveals both hands and the crib.
_PUBLIC = frozenset({"cut", "heels", "play", "reset", "go", "hand", "crib", "win"})


@dataclass(frozen=True)
class VisibleEvent:
    """One line of history, already redacted for the player who will see it."""

    round: int
    kind: str
    actor: Optional[int]
    points: int
    detail: str
    cards: tuple[int, ...] = ()

    def is_mine(self, player: int) -> bool:
        return self.actor == player


def visible_events(state: CribbageState, player: int) -> list[VisibleEvent]:
    """The event log with everything ``player`` may not see stripped out."""
    visible: list[VisibleEvent] = []
    for event in state.events:
        kind = event.kind

        if kind == "deal":
            # Carries both hands.  Keep only that a round began, and whose deal.
            visible.append(
                VisibleEvent(event.round, kind, event.player, 0, "")
            )
            continue

        if kind == "discard":
            mine = event.player == player
            cards = tuple(state.discards[player]) if mine else ()
            visible.append(
                VisibleEvent(
                    event.round, kind, event.player, 0,
                    hand_str(cards) if mine else "", cards,
                )
            )
            continue

        if kind in _PUBLIC:
            # A reset belongs to nobody: attributing it to the last player to
            # lay a card reads as if they did something.
            actor = None if kind == "reset" else event.player
            visible.append(
                VisibleEvent(event.round, kind, actor, event.points, event.detail)
            )
            continue

        # Anything unrecognised is withheld rather than leaked by default.
    return visible


def card_json(card: int) -> dict:
    """A card as the front end wants it: id, name, and the parts to style by."""
    return {
        "id": card,
        "name": card_str(card),
        "rank": rank_of(card),
        "suit": suit_of(card),
        "red": suit_of(card) in (1, 2),
    }


def game_view(state: CribbageState, player: int) -> dict:
    """A JSON-ready snapshot of the game from ``player``'s side.

    Contains their hand, their own lay-away, everything face up, and the legal
    actions -- and nothing else.  Built from
    :meth:`~cribbage.engine.CribbageState.information_state`, so it inherits the
    same guarantees the engine's tests assert.
    """
    opponent = 1 - player
    info = state.information_state(player)
    terminal = state.is_terminal()

    legal: list = []
    if not terminal and state.current_player == player:
        legal = [
            list(action) if isinstance(action, tuple) else [action]
            for action in info.legal
        ]

    view = {
        "phase": ("over" if terminal else
                  "discard" if info.phase is Phase.DISCARD else "play"),
        "round": info.round_index,
        "target": info.target,
        "dealer": info.dealer,
        "isDealer": info.is_dealer,
        "yourTurn": (not terminal) and state.current_player == player,
        "scores": {"you": info.my_score, "opponent": info.opp_score},
        "hand": [card_json(card) for card in info.hand],
        "yourLayAway": [card_json(card) for card in info.my_discards],
        "starter": card_json(info.starter) if info.starter is not None else None,
        "count": info.count,
        "sequence": [card_json(card) for card in info.seq],
        "yourPlayed": [card_json(card) for card in info.my_played],
        "opponentPlayed": [card_json(card) for card in info.opp_played],
        "opponentHandSize": info.opp_hand_size,
        "legal": legal,
        "events": [
            {
                "round": event.round,
                "kind": event.kind,
                "actor": ("you" if event.actor == player
                          else "opponent" if event.actor == opponent else None),
                "points": event.points,
                "detail": event.detail,
                "cards": [card_json(card) for card in event.cards],
            }
            for event in visible_events(state, player)
        ],
        "result": None,
        "cutLuck": None,
    }

    if terminal:
        result = state.result()
        luck: CutLuck = cut_luck(state, player)
        view["result"] = {
            "youWon": result.winner == player,
            "scores": {"you": result.scores[player],
                       "opponent": result.scores[opponent]},
            "margin": result.margin,
            "skunk": result.skunk,
            "doubleSkunk": result.double_skunk,
            "rounds": result.rounds,
        }
        view["cutLuck"] = {
            "net": round(luck.net, 2),
            "sd": round(luck.net_sd, 2),
            "z": round(luck.net_z, 2),
            "gross": round(luck.gross, 2),
            "rounds": len(luck.rounds),
            "skipped": luck.skipped,
            "summary": luck.describe(),
            "perRound": [
                {
                    "round": row.round_index,
                    "starter": card_json(row.starter),
                    "gross": round(row.gross, 2),
                    "net": round(row.net, 2),
                    "sd": round(row.sd_net, 2),
                    "z": round(row.net_z, 2),
                }
                for row in luck.rounds
            ],
        }
    return view
