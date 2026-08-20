"""The rules engine: a two-player, 121-point cribbage state machine.

The shape here is deliberately OpenSpiel-like -- ``legal_actions`` /
``apply_action`` / ``clone`` -- rather than a loop that calls into agent
callbacks.  Callbacks are easier to write but dead-end for search: a tree search
needs to fork a position and drive it itself, which an inverted-control loop
cannot offer.

Two conventions make the rest of the code simpler:

* **Chance is internal.**  The deal and the cut are resolved by the state's own
  RNG rather than exposed as chance nodes, and :meth:`CribbageState.apply_action`
  fast-forwards through every forced transition -- gos, count resets, the show,
  the next deal.  A caller therefore never sees a state that is not either a
  real decision or terminal, and never has to ask whether something automatic is
  pending.
* **Discard is sequential.**  Real cribbage discards simultaneously; this models
  pone-then-dealer.  That is only sound because the dealer's information state
  says nothing about *which* cards the pone laid away -- see
  :meth:`CribbageState.information_state`.

Actions are native game objects, not indices: a discard is a ``tuple`` of two
card ints, a play is a single card int.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations
from typing import Optional, Sequence

from .cards import CARD_VALUES, JACK, NUM_CARDS, card_str, hand_str
from .scoring import score_hand_detail, score_play

__all__ = [
    "Phase",
    "Event",
    "InfoState",
    "GameResult",
    "CribbageState",
    "new_game",
    "DEFAULT_TARGET",
    "SKUNK_LINE",
    "DOUBLE_SKUNK_LINE",
]

DEFAULT_TARGET = 121
SKUNK_LINE = 91
DOUBLE_SKUNK_LINE = 61

HAND_SIZE = 6
KEEP_SIZE = 4
DISCARD_SIZE = 2
MAX_COUNT = 31


class Phase(IntEnum):
    DISCARD = 0
    PLAY = 1
    GAME_OVER = 2


#: Event kinds that carry points, for statistics aggregation.
POINT_KINDS = ("heels", "play", "go", "hand", "crib")


@dataclass(frozen=True)
class Event:
    """One line of the game's history.

    ``kind`` is one of ``deal``, ``discard``, ``cut``, ``heels``, ``play``,
    ``reset``, ``go``, ``hand``, ``crib``, ``win``.  Zero-point events are kept
    because they are what makes a transcript readable and what lets a test
    recompute the play independently.
    """

    round: int
    player: int
    points: int
    kind: str
    detail: str = ""


@dataclass(frozen=True)
class InfoState:
    """Everything one player legally knows, and nothing else.

    This is the only view an agent is given.  In particular it never contains
    the opponent's hand, the opponent's crib discards, or the undealt remainder
    of the deck.
    """

    player: int
    phase: Phase
    dealer: int
    round_index: int
    target: int

    my_score: int
    opp_score: int

    hand: tuple[int, ...]
    my_discards: tuple[int, ...]
    starter: Optional[int]

    count: int
    seq: tuple[int, ...]
    play_order: tuple[tuple[int, int], ...]
    my_played: tuple[int, ...]
    opp_played: tuple[int, ...]
    opp_hand_size: int

    legal: tuple

    @property
    def is_dealer(self) -> bool:
        return self.player == self.dealer

    @property
    def seen(self) -> frozenset[int]:
        """Cards whose location this player already knows."""
        cards = set(self.hand) | set(self.my_discards) | set(self.my_played)
        cards |= set(self.opp_played)
        if self.starter is not None:
            cards.add(self.starter)
        return frozenset(cards)

    @property
    def unseen(self) -> tuple[int, ...]:
        """Cards that could still be anywhere the player cannot see."""
        seen = self.seen
        return tuple(c for c in range(NUM_CARDS) if c not in seen)


@dataclass(frozen=True)
class GameResult:
    winner: int
    scores: tuple[int, int]
    rounds: int

    @property
    def loser(self) -> int:
        return 1 - self.winner

    @property
    def margin(self) -> int:
        return self.scores[self.winner] - self.scores[self.loser]

    @property
    def skunk(self) -> bool:
        """Loser failed to reach the skunk line (91)."""
        return self.scores[self.loser] < SKUNK_LINE

    @property
    def double_skunk(self) -> bool:
        return self.scores[self.loser] < DOUBLE_SKUNK_LINE


class CribbageState:
    """A complete cribbage position, including the hidden information."""

    __slots__ = (
        "target", "scores", "dealer", "phase", "round_index",
        "hands", "dealt", "kept", "discards", "crib", "starter", "deck",
        "count", "seq", "played", "play_order", "turn", "last_to_play",
        "discard_turn", "events", "winner", "_rng",
    )

    def __init__(
        self,
        dealer: int = 0,
        target: int = DEFAULT_TARGET,
        seed: Optional[int] = None,
        rng: Optional[random.Random] = None,
        _blank: bool = False,
    ):
        self.target = target
        self.scores = [0, 0]
        self.dealer = dealer
        self.round_index = 0
        self.winner: Optional[int] = None
        self.events: list[Event] = []
        self._rng = rng if rng is not None else random.Random(seed)

        self.phase = Phase.DISCARD
        self.hands: list[list[int]] = [[], []]
        self.dealt: list[list[int]] = [[], []]
        self.kept: list[list[int]] = [[], []]
        self.discards: list[list[int]] = [[], []]
        self.crib: list[int] = []
        self.starter: Optional[int] = None
        self.deck: list[int] = []
        self.count = 0
        self.seq: list[int] = []
        self.played: list[list[int]] = [[], []]
        self.play_order: list[tuple[int, int]] = []
        self.turn = 1 - dealer
        self.last_to_play: Optional[int] = None
        self.discard_turn = 1 - dealer

        if not _blank:
            self._begin_round()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def pone(self) -> int:
        """The non-dealer, who leads the play and counts first at the show."""
        return 1 - self.dealer

    def is_terminal(self) -> bool:
        return self.phase is Phase.GAME_OVER

    @property
    def current_player(self) -> Optional[int]:
        """Whose decision it is, or ``None`` if the game is over.

        Never returns a player who has no legal action: a player who cannot
        play during the play phase is skipped by the engine rather than asked.
        """
        if self.phase is Phase.GAME_OVER:
            return None
        return self.discard_turn if self.phase is Phase.DISCARD else self.turn

    def legal_actions(self) -> list:
        """Legal actions for :attr:`current_player`.

        During ``DISCARD`` these are the 15 two-card tuples; during ``PLAY``
        they are the cards that fit under 31.  Never empty unless terminal.
        """
        if self.phase is Phase.GAME_OVER:
            return []
        if self.phase is Phase.DISCARD:
            return list(combinations(sorted(self.hands[self.discard_turn]), DISCARD_SIZE))
        room = MAX_COUNT - self.count
        return [c for c in self.hands[self.turn] if CARD_VALUES[c] <= room]

    def result(self) -> GameResult:
        if self.winner is None:
            raise ValueError("game is not over")
        return GameResult(
            winner=self.winner,
            scores=(self.scores[0], self.scores[1]),
            rounds=self.round_index,
        )

    def returns(self) -> tuple[float, float]:
        """Zero-sum outcome from each seat: +1 for the winner, -1 for the loser."""
        if self.winner is None:
            return (0.0, 0.0)
        return (1.0, -1.0) if self.winner == 0 else (-1.0, 1.0)

    def information_state(self, player: int) -> InfoState:
        """The view of the position that ``player`` is entitled to.

        Deliberately excludes ``self.crib`` and the opponent's hand.  During the
        discard phase the dealer sees their own two laid-away cards but nothing
        about the pone's, which is what makes the sequential model of a
        simultaneous discard sound.  The opponent's *hand size* is visible, but
        that is public information at the table too.
        """
        opp = 1 - player
        return InfoState(
            player=player,
            phase=self.phase,
            dealer=self.dealer,
            round_index=self.round_index,
            target=self.target,
            my_score=self.scores[player],
            opp_score=self.scores[opp],
            hand=tuple(sorted(self.hands[player])),
            my_discards=tuple(self.discards[player]),
            starter=self.starter,
            count=self.count,
            seq=tuple(self.seq),
            play_order=tuple(self.play_order),
            my_played=tuple(self.played[player]),
            opp_played=tuple(self.played[opp]),
            opp_hand_size=len(self.hands[opp]),
            legal=tuple(self.legal_actions()) if self.current_player == player else (),
        )

    # ------------------------------------------------------------------
    # Copying and determinization
    # ------------------------------------------------------------------

    def clone(self) -> "CribbageState":
        """A deep copy, RNG state included.

        The RNG is copied exactly rather than reseeded, so a clone replays
        identically.  Search code that wants rollouts to diverge should pass its
        own ``random.Random`` to :meth:`determinize` or call :meth:`reseed`.
        """
        other = CribbageState.__new__(CribbageState)
        other.target = self.target
        other.scores = list(self.scores)
        other.dealer = self.dealer
        other.phase = self.phase
        other.round_index = self.round_index
        other.hands = [list(self.hands[0]), list(self.hands[1])]
        other.dealt = [list(self.dealt[0]), list(self.dealt[1])]
        other.kept = [list(self.kept[0]), list(self.kept[1])]
        other.discards = [list(self.discards[0]), list(self.discards[1])]
        other.crib = list(self.crib)
        other.starter = self.starter
        other.deck = list(self.deck)
        other.count = self.count
        other.seq = list(self.seq)
        other.played = [list(self.played[0]), list(self.played[1])]
        other.play_order = list(self.play_order)
        other.turn = self.turn
        other.last_to_play = self.last_to_play
        other.discard_turn = self.discard_turn
        other.events = list(self.events)
        other.winner = self.winner
        other._rng = random.Random()
        other._rng.setstate(self._rng.getstate())
        return other

    def reseed(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def determinize(self, player: int, rng: Optional[random.Random] = None) -> "CribbageState":
        """Sample a full world consistent with ``player``'s information state.

        This is the ISMCTS primitive: the returned state is a legal, complete
        position that ``player`` cannot distinguish from the real one.  Their own
        hand, their own laid-away cards, the starter and every card played face
        up are preserved exactly; the opponent's hand, the opponent's crib
        contribution and the undealt remainder are redealt at their known sizes
        from the cards ``player`` has not seen.
        """
        rng = rng if rng is not None else self._rng
        opp = 1 - player
        clone = self.clone()

        seen = set(self.hands[player]) | set(self.discards[player])
        seen |= set(self.played[0]) | set(self.played[1])
        if self.starter is not None:
            seen.add(self.starter)

        pool = [c for c in range(NUM_CARDS) if c not in seen]
        rng.shuffle(pool)

        take = len(self.hands[opp])
        clone.hands[opp] = sorted(pool[:take])
        cursor = take

        take = len(self.discards[opp])
        clone.discards[opp] = pool[cursor : cursor + take]
        cursor += take

        clone.deck = pool[cursor:]
        clone.crib = list(clone.discards[0]) + list(clone.discards[1])
        clone.dealt[opp] = sorted(
            clone.hands[opp] + clone.discards[opp] + clone.played[opp]
        )
        if clone.kept[opp]:
            clone.kept[opp] = sorted(clone.hands[opp] + clone.played[opp])
        return clone

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def apply_action(self, action) -> None:
        """Apply one decision and fast-forward to the next decision or terminal."""
        if self.phase is Phase.GAME_OVER:
            raise ValueError("game is over")
        if self.phase is Phase.DISCARD:
            self._apply_discard(action)
        else:
            self._apply_play(action)

    def _apply_discard(self, action) -> None:
        player = self.discard_turn
        try:
            cards = tuple(action)
        except TypeError:
            raise ValueError(f"discard action must be two cards, got {action!r}") from None
        if len(cards) != DISCARD_SIZE or len(set(cards)) != DISCARD_SIZE:
            raise ValueError(f"discard must be two distinct cards, got {cards!r}")
        for c in cards:
            if not isinstance(c, int) or not 0 <= c < NUM_CARDS:
                raise ValueError(f"not a card: {c!r}")
        hand = self.hands[player]
        for c in cards:
            if c not in hand:
                raise ValueError(f"player {player} cannot discard {card_str(c)}: not held")

        for c in cards:
            hand.remove(c)
        self.discards[player] = list(cards)
        self.kept[player] = sorted(hand)
        self.crib.extend(cards)
        self.events.append(
            Event(self.round_index, player, 0, "discard", hand_str(cards))
        )

        if player == self.pone:
            self.discard_turn = self.dealer
        else:
            self._cut_and_start_play()

    def _apply_play(self, action) -> None:
        if not isinstance(action, int) or not 0 <= action < NUM_CARDS:
            raise ValueError(f"play action must be a single card int, got {action!r}")
        if action not in self.hands[self.turn]:
            raise ValueError(
                f"player {self.turn} cannot play {card_str(action)}: not held"
            )
        if self.count + CARD_VALUES[action] > MAX_COUNT:
            raise ValueError(
                f"playing {card_str(action)} would take the count past {MAX_COUNT}"
            )
        self._play_card(action)
        if self.phase is Phase.GAME_OVER:
            return
        self._advance_play()

    # ------------------------------------------------------------------
    # Round structure
    # ------------------------------------------------------------------

    def _begin_round(self) -> None:
        self.round_index += 1
        deck = list(range(NUM_CARDS))
        self._rng.shuffle(deck)
        self.hands = [sorted(deck[:HAND_SIZE]), sorted(deck[HAND_SIZE : 2 * HAND_SIZE])]
        self.deck = deck[2 * HAND_SIZE :]
        self.dealt = [list(self.hands[0]), list(self.hands[1])]
        self.kept = [[], []]
        self.discards = [[], []]
        self.crib = []
        self.starter = None
        self.count = 0
        self.seq = []
        self.played = [[], []]
        self.play_order = []
        self.last_to_play = None
        self.turn = self.pone
        self.discard_turn = self.pone
        self.phase = Phase.DISCARD
        self.events.append(
            Event(
                self.round_index, self.dealer, 0, "deal",
                f"P0 {hand_str(self.hands[0])} | P1 {hand_str(self.hands[1])}",
            )
        )

    def _cut_and_start_play(self) -> None:
        """Turn the starter, pay his heels, and open the play."""
        self.starter = self.deck.pop(self._rng.randrange(len(self.deck)))
        self.events.append(
            Event(self.round_index, self.dealer, 0, "cut", card_str(self.starter))
        )
        if (self.starter >> 2) == JACK:
            self._award(self.dealer, 2, "heels", "his heels")
            if self.phase is Phase.GAME_OVER:
                return

        self.phase = Phase.PLAY
        self.count = 0
        self.seq = []
        self.played = [[], []]
        self.play_order = []
        self.turn = self.pone
        self.last_to_play = None
        self._advance_play()

    def _play_card(self, card: int) -> None:
        player = self.turn
        self.hands[player].remove(card)
        self.played[player].append(card)
        self.play_order.append((player, card))
        self.seq.append(card)
        self.count += CARD_VALUES[card]
        self.last_to_play = player

        points = score_play(self.seq)
        self._award(
            player, points, "play", f"{card_str(card)} for {self.count}"
        )
        if self.phase is Phase.GAME_OVER:
            return

        if self.count == MAX_COUNT:
            # Thirty-one already paid 2 through score_play.  Resetting here is
            # what stops the go branch from paying a second point for it.
            self._reset_subround()
            self.turn = 1 - player
        else:
            self.turn = 1 - player

    def _reset_subround(self) -> None:
        self.events.append(
            Event(self.round_index, self.last_to_play or 0, 0, "reset", "count resets")
        )
        self.count = 0
        self.seq = []

    def _advance_play(self) -> None:
        """Run the play forward until someone has a real decision, or it ends.

        This is the loop that handles gos.  A go is not a turn change: when one
        player cannot play, the other keeps laying cards -- scoring pairs and
        runs off their own sequence -- until neither can.
        """
        while True:
            if not self.hands[0] and not self.hands[1]:
                # Both out of cards.  A non-empty seq means the final card did
                # not make 31, so the last card played is worth a point.
                if self.seq and self.last_to_play is not None:
                    self._award(self.last_to_play, 1, "go", "last card")
                    if self.phase is Phase.GAME_OVER:
                        return
                self._show()
                return

            if self._can_play(self.turn):
                return  # a real decision

            other = 1 - self.turn
            if self._can_play(other):
                self.turn = other
                continue

            # Neither can play and cards remain: go.  The count cannot be 31
            # here, because hitting 31 resets immediately in _play_card.
            if self.last_to_play is None:
                raise AssertionError("go with no card played")
            self._award(self.last_to_play, 1, "go", "go")
            if self.phase is Phase.GAME_OVER:
                return
            leader = 1 - self.last_to_play
            self._reset_subround()
            self.turn = leader

    def _can_play(self, player: int) -> bool:
        room = MAX_COUNT - self.count
        return any(CARD_VALUES[c] <= room for c in self.hands[player])

    def _show(self) -> None:
        """Count the hands, then the crib -- pone first, and stop the instant
        someone crosses the target.  The order is load-bearing: a dealer can be
        holding a winning crib and still lose to the pone's hand."""
        starter = self.starter
        assert starter is not None

        for player in (self.pone, self.dealer):
            hand = self.kept[player]
            detail = score_hand_detail(hand, starter)
            self._award(
                player,
                detail.total,
                "hand",
                f"{hand_str(hand)} + {card_str(starter)}: {detail.describe()}",
            )
            if self.phase is Phase.GAME_OVER:
                return

        crib_detail = score_hand_detail(self.crib, starter, is_crib=True)
        self._award(
            self.dealer,
            crib_detail.total,
            "crib",
            f"{hand_str(self.crib)} + {card_str(starter)}: {crib_detail.describe()}",
        )
        if self.phase is Phase.GAME_OVER:
            return

        self.dealer = 1 - self.dealer
        self._begin_round()

    def _award(self, player: int, points: int, kind: str, detail: str = "") -> None:
        """The single choke point for scoring, so the win check cannot be missed."""
        if points:
            self.scores[player] += points
        self.events.append(Event(self.round_index, player, points, kind, detail))
        if self.scores[player] >= self.target:
            self.winner = player
            self.phase = Phase.GAME_OVER
            self.events.append(
                Event(
                    self.round_index, player, 0, "win",
                    f"player {player} wins {self.scores[player]}-{self.scores[1 - player]}",
                )
            )

    # ------------------------------------------------------------------
    # Test / analysis construction
    # ------------------------------------------------------------------

    @classmethod
    def for_play(
        cls,
        hands: Sequence[Sequence[int]],
        starter: int,
        dealer: int = 0,
        crib: Optional[Sequence[int]] = None,
        scores: Sequence[int] = (0, 0),
        target: int = DEFAULT_TARGET,
        seed: Optional[int] = None,
    ) -> "CribbageState":
        """Build a state sitting at the start of the play with exact hands.

        For tests and analysis.  The discard is skipped and his heels is *not*
        paid even if the starter is a jack, so a scripted position scores only
        what the caller intends.
        """
        state = cls(dealer=dealer, target=target, seed=seed, _blank=True)
        state.scores = list(scores)
        state.round_index = 1
        state.hands = [sorted(hands[0]), sorted(hands[1])]
        state.kept = [sorted(hands[0]), sorted(hands[1])]
        state.dealt = [sorted(hands[0]), sorted(hands[1])]

        used = set(state.hands[0]) | set(state.hands[1]) | {starter}
        remaining = [c for c in range(NUM_CARDS) if c not in used]
        if crib is None:
            crib = remaining[:4]
        state.crib = list(crib)
        state.discards = [list(crib[:2]), list(crib[2:])]
        state.deck = [c for c in remaining if c not in set(crib)]

        state.starter = starter
        state.phase = Phase.PLAY
        state.count = 0
        state.seq = []
        state.played = [[], []]
        state.play_order = []
        state.turn = state.pone
        state.last_to_play = None
        state._advance_play()
        return state


def new_game(
    seed: Optional[int] = None,
    dealer: int = 0,
    target: int = DEFAULT_TARGET,
) -> CribbageState:
    """A fresh game, dealt and sitting at the pone's discard decision."""
    return CribbageState(dealer=dealer, target=target, seed=seed)
