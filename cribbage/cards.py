"""Card representation.

A card is a plain ``int`` in ``[0, 52)``.  There is no ``Card`` class, and that is
deliberate::

    rank = card >> 2      # 0 = ace, 1 = two, ... 9 = ten, 10 = jack, 11 = queen, 12 = king
    suit = card & 3       # 0 = spades, 1 = hearts, 2 = diamonds, 3 = clubs

Three properties fall out of this encoding and all three get used:

* Hot paths stay allocation-free -- scoring a hand touches no Python objects
  beyond the ints themselves.
* ``sorted(cards)`` sorts by rank, which is what run detection wants.
* A card doubles as an index into a 52-wide array, so a future neural net's
  feature planes need no translation step.

Note the split between *rank* and *value*: runs care about rank (a king is 12,
one above a queen), while fifteens and the running count in the play care about
value (a king is worth 10, same as a ten).
"""

from __future__ import annotations

from typing import Iterable, Sequence

__all__ = [
    "RANK_CHARS",
    "SUIT_CHARS",
    "SUIT_SYMBOLS",
    "NUM_CARDS",
    "JACK",
    "rank_of",
    "suit_of",
    "value_of",
    "make_card",
    "make_deck",
    "card_str",
    "hand_str",
    "parse_card",
    "parse_hand",
    "hand_value",
]

RANK_CHARS = "A23456789TJQK"
SUIT_CHARS = "SHDC"
SUIT_SYMBOLS = "♠♥♦♣"

NUM_CARDS = 52
NUM_RANKS = 13
NUM_SUITS = 4

#: Rank index of a jack, needed by his-heels and nobs.
JACK = 10

#: ``CARD_VALUES[c]`` is the counting value of card ``c`` (ace 1, face cards 10).
CARD_VALUES = tuple(min((c >> 2) + 1, 10) for c in range(NUM_CARDS))

#: ``RANK_VALUES[r]`` is the counting value of rank index ``r``.
RANK_VALUES = tuple(min(r + 1, 10) for r in range(NUM_RANKS))


def rank_of(card: int) -> int:
    """Rank index of ``card``: 0 for an ace through 12 for a king."""
    return card >> 2


def suit_of(card: int) -> int:
    """Suit index of ``card``: 0=spades, 1=hearts, 2=diamonds, 3=clubs."""
    return card & 3


def value_of(card: int) -> int:
    """Counting value of ``card``: ace 1, pip cards their pips, face cards 10."""
    return CARD_VALUES[card]


def make_card(rank: int, suit: int) -> int:
    """Build a card int from a rank index and a suit index."""
    return (rank << 2) | suit


def make_deck() -> list[int]:
    """A fresh ordered deck of all 52 cards."""
    return list(range(NUM_CARDS))


def hand_value(cards: Iterable[int]) -> int:
    """Total counting value of ``cards`` -- the running count in the play."""
    return sum(CARD_VALUES[c] for c in cards)


def card_str(card: int, symbols: bool = False) -> str:
    """Render a card as e.g. ``"5H"``, or ``"5♥"`` when ``symbols`` is set."""
    suits = SUIT_SYMBOLS if symbols else SUIT_CHARS
    return f"{RANK_CHARS[card >> 2]}{suits[card & 3]}"


def hand_str(cards: Iterable[int], symbols: bool = False) -> str:
    """Render a collection of cards, sorted, space separated."""
    return " ".join(card_str(c, symbols) for c in sorted(cards))


def parse_card(text: str) -> int:
    """Parse ``"5H"`` / ``"5h"`` / ``"TS"`` into a card int.

    Raises ``ValueError`` on anything that is not exactly a rank char followed
    by a suit char.
    """
    text = text.strip().upper()
    if len(text) != 2:
        raise ValueError(f"bad card {text!r}: expected two characters like '5H'")
    rank = RANK_CHARS.find(text[0])
    suit = SUIT_CHARS.find(text[1])
    if rank < 0:
        raise ValueError(f"bad rank {text[0]!r} in {text!r}: expected one of {RANK_CHARS}")
    if suit < 0:
        raise ValueError(f"bad suit {text[1]!r} in {text!r}: expected one of {SUIT_CHARS}")
    return (rank << 2) | suit


def parse_hand(text: str | Sequence[str]) -> list[int]:
    """Parse ``"5H 5S 5D JC"`` (or a sequence of card strings) into card ints.

    Rejects duplicates, since every caller of this is building a legal deal.
    """
    parts = text.split() if isinstance(text, str) else list(text)
    cards = [parse_card(p) for p in parts]
    if len(set(cards)) != len(cards):
        raise ValueError(f"duplicate cards in {text!r}")
    return cards
