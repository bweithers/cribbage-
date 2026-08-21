"""Expected pegging value of a four-card keep.

This closes the largest known gap in :class:`~cribbage.agents.heuristic.HeuristicAgent`.
That agent chooses a discard by maximizing hand score plus or minus the crib,
and is completely blind to how the cards it keeps will *play*.  Four fives are
worth twenty at the show and peg abominably; A-2-3-4 shows six and pegs
beautifully.  Nothing in an expected-hand-score calculation can tell them apart.

Pegging depends only on rank -- runs and pairs use rank, fifteens and the count
use value, and suits never enter -- so the table is keyed on the four-card rank
multiset.  There are ``C(16,4) = 1820`` of those, all reachable, which is small
enough to tabulate.  Two tables are kept, because the deal changes pegging
materially: the dealer scores about 3.4 non-hand points a deal against the
pone's 2.0, and holds the last card.

Each entry is estimated by simulating the play, so unlike
:mod:`cribbage.crib_table` it is sampled rather than exact.  Two things keep the
noise down:

* **A realistic opponent.**  The opposing six are dealt and then *discarded* by
  the reference agent, rather than four cards being drawn at random.  Real
  opponents keep good hands, and pegging against a real holding is not the same
  problem as pegging against a random one.
* **Common random numbers.**  Scenario *i* uses ``Random(i)`` for every entry in
  the table, so the same shuffles face each candidate keep.  What matters here
  is the *difference* between keeps, and differences estimated under shared
  randomness are far tighter than independent ones.
"""

from __future__ import annotations

import json
import random
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Optional, Sequence

from .cards import NUM_CARDS
from .engine import CribbageState, Phase

__all__ = [
    "DATA_FILE", "peg_ev", "load_peg_table", "table_key", "cards_for_ranks",
    "simulate_pegging", "estimate_entry", "all_keys", "reference_discard",
]

DATA_FILE = "peg_ev.json"
KEEP_SIZE = 4

_TABLE: Optional[dict[str, list[float]]] = None


def table_key(ranks: Sequence[int]) -> str:
    """Canonical key for a keep: its sorted rank multiset."""
    return ",".join(str(rank) for rank in sorted(ranks))


def all_keys() -> list[tuple[int, ...]]:
    """Every four-card rank multiset that can be held from one deck."""
    return [
        ranks
        for ranks in combinations_with_replacement(range(13), KEEP_SIZE)
        if max(ranks.count(rank) for rank in set(ranks)) <= 4
    ]


def cards_for_ranks(ranks: Sequence[int]) -> list[int]:
    """Concrete cards with those ranks.  Suits are arbitrary: pegging ignores them."""
    used: dict[int, int] = {}
    cards = []
    for rank in ranks:
        suit = used.get(rank, 0)
        used[rank] = suit + 1
        cards.append((rank << 2) | suit)
    return cards


def reference_discard(agent, six: Sequence[int], is_dealer: bool) -> Sequence[int]:
    """What the reference agent would lay away from these six.

    Used to give the simulated opponent a realistic holding.  Builds the
    information state by hand because there is no game around this decision --
    only a hypothetical hand.
    """
    from itertools import combinations

    from .engine import InfoState, Phase

    hand = tuple(sorted(six))
    info = InfoState(
        player=0, phase=Phase.DISCARD, dealer=0 if is_dealer else 1,
        round_index=1, target=121, my_score=0, opp_score=0, hand=hand,
        my_discards=(), starter=None, count=0, seq=(), play_order=(),
        my_played=(), opp_played=(), opp_hand_size=6,
        legal=tuple(combinations(hand, 2)),
    )
    return agent.discard(info)


def simulate_pegging(keep: Sequence[int], opponent_keep: Sequence[int],
                     starter: int, is_dealer: bool, agent) -> int:
    """Play out one pegging phase and return my points minus theirs.

    Only the play is scored: the show is what the rest of the discard objective
    already accounts for.
    """
    me = 0
    dealer = 0 if is_dealer else 1
    hands = [list(keep), list(opponent_keep)]
    state = CribbageState.for_play(hands=hands, starter=starter, dealer=dealer)

    while state.phase is Phase.PLAY:
        player = state.current_player
        assert player is not None
        state.apply_action(agent.play(state.information_state(player)))

    pegged = [0, 0]
    for event in state.events:
        if event.round == 1 and event.kind in ("play", "go"):
            pegged[event.player] += event.points
    return pegged[me] - pegged[1 - me]


def estimate_entry(ranks: Sequence[int], is_dealer: bool, samples: int,
                   agent, discarder) -> float:
    """Mean pegging differential for a keep, over ``samples`` shared scenarios."""
    keep = cards_for_ranks(ranks)
    held = set(keep)
    total = 0
    for scenario in range(samples):
        rng = random.Random(scenario)  # shared across every entry: see module docs
        deck = [card for card in range(NUM_CARDS) if card not in held]
        rng.shuffle(deck)

        # Deal the opponent six and let the reference agent choose what to keep,
        # so the pegging is against a real holding rather than random cards.
        their_six = sorted(deck[:6])
        starter = deck[6]
        laid = discarder(their_six, not is_dealer)
        their_keep = [card for card in their_six if card not in set(laid)]

        total += simulate_pegging(keep, their_keep, starter, is_dealer, agent)
    return total / samples


def _data_path() -> Path:
    return Path(__file__).parent / "data" / DATA_FILE


def load_peg_table() -> dict[str, list[float]]:
    """The shipped table: key -> [as pone, as dealer]."""
    global _TABLE
    if _TABLE is None:
        path = _data_path()
        if not path.exists():  # pragma: no cover - only without the data file
            raise FileNotFoundError(
                f"{path} is missing; run scripts/build_peg_table.py to generate it"
            )
        _TABLE = json.loads(path.read_text())
    return _TABLE


def peg_ev(ranks: Sequence[int], is_dealer: bool) -> float:
    """Expected pegging points minus the opponent's, for a keep of those ranks."""
    return load_peg_table()[table_key(ranks)][1 if is_dealer else 0]
