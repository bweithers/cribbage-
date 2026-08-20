"""A one-ply expected-value agent: the baseline any future net has to beat.

Both decisions are expectations over the cards the agent cannot see, which makes
this an honest opponent rather than a bundle of maxims.

**Discard.**  For each of the 15 ways to keep four cards, the expected hand score
is computed *exactly* -- every one of the 46 possible starters is enumerated and
scored.  That is affordable only because hand scoring is a table lookup.  The
crib is then added if the agent is dealer and subtracted if it is not, using the
expectation from :mod:`cribbage.crib_table`.

**Play.**  For each legal card: the points it scores now, minus what the
opponent is expected to score in reply, plus the go point when the card is
likely to leave the opponent unable to answer.  The reply term is averaged over
the cards the opponent could still be holding, and the probability that they
hold *any* playable card is computed hypergeometrically from the size of their
hand.  Familiar cribbage maxims -- do not lead a five, do not take the count to
21 -- are not coded anywhere; they fall out of that reply term.

Known gaps, so nothing is measured against this under the wrong impression.  The
46-starter enumeration is exact; the policy around it is not:

* The crib term assumes the opponent lays away uniformly at random.
* The crib table is keyed on the lay-away alone, ignoring the rest of your hand.
* The discard ignores pegging value completely -- four fives peg badly and
  A-2-3-4 pegs well, and nothing here can tell the difference.
* It maximizes points, not win probability.  ``my_score`` and ``opp_score`` are
  right there in the information state and are never read, so it plays 118-115
  exactly as it plays 20-15.

That makes it a floor to measure against rather than a ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Sequence

from ..cards import CARD_VALUES, NUM_CARDS
from ..crib_table import crib_ev
from ..engine import MAX_COUNT, InfoState
from ..scoring import score_hand, score_play
from .base import Agent

__all__ = ["HeuristicAgent", "DiscardOption"]


@dataclass(frozen=True)
class DiscardOption:
    """One of the fifteen ways to split a dealt hand, fully evaluated.

    ``scores`` is the hand's score against every one of the 46 possible
    starters, in starter order.  Keeping the whole list rather than just its
    mean costs nothing -- the enumeration has already happened -- and it is what
    lets a position-aware agent ask for ``P(score >= n)`` instead of settling
    for the average.
    """

    pair: tuple[int, int]
    keep: tuple[int, ...]
    scores: tuple[int, ...]
    crib: float

    @property
    def mean(self) -> float:
        return sum(self.scores) / len(self.scores)

    @property
    def stdev(self) -> float:
        """Spread of the hand's score across starters.

        Free, since the enumeration already happened, and it is what lets an
        agent prefer a safe keep when ahead and a swingy one when behind.
        """
        mean = self.mean
        return (sum((s - mean) ** 2 for s in self.scores) / len(self.scores)) ** 0.5

    def probability_at_least(self, needed: float) -> float:
        """Share of starters that bring the hand to ``needed`` points or more."""
        if needed <= 0:
            return 1.0
        return sum(1 for s in self.scores if s >= needed) / len(self.scores)


class HeuristicAgent(Agent):
    name = "heuristic"

    def __init__(self, crib_weight: float = 1.0, risk_weight: float = 1.0):
        #: Scales the crib term in the discard.  1.0 trusts the table fully.
        self.crib_weight = crib_weight
        #: Scales the opponent's expected reply in the play.  0 plays greedily.
        self.risk_weight = risk_weight

    # ------------------------------------------------------------------

    def discard_options(self, info: InfoState) -> list[DiscardOption]:
        """Evaluate all fifteen lay-aways against all 46 possible starters.

        Shared with :class:`~cribbage.agents.positional.PositionalAgent`, which
        keeps the expensive enumeration in one place and lets the subclass change
        only the objective applied to it.
        """
        hand = list(info.hand)
        held = set(hand)
        starters = [c for c in range(NUM_CARDS) if c not in held]

        options = []
        for pair in combinations(hand, 2):
            laid = set(pair)
            keep = tuple(c for c in hand if c not in laid)
            options.append(
                DiscardOption(
                    pair=pair,
                    keep=keep,
                    scores=tuple(score_hand(keep, s) for s in starters),
                    crib=crib_ev(pair[0], pair[1]),
                )
            )
        return options

    def discard_value(self, option: DiscardOption, info: InfoState) -> float:
        """Expected points this deal: my hand, plus or minus the crib."""
        crib = option.crib * self.crib_weight
        return option.mean + (crib if info.is_dealer else -crib)

    def discard(self, info: InfoState) -> Sequence[int]:
        options = self.discard_options(info)
        best = max(options, key=lambda option: self.discard_value(option, info))
        return best.pair

    # ------------------------------------------------------------------

    def effective_risk(self, info: InfoState) -> float:
        """How much to fear the opponent's reply.  Constant here; position-aware
        subclasses scale it by who is closer to going out."""
        return self.risk_weight

    def play(self, info: InfoState) -> int:
        legal = list(info.legal)
        if len(legal) == 1:
            return legal[0]

        risk_weight = self.effective_risk(info)
        seq = list(info.seq)
        # Cards the opponent might hold.  This pool also contains the crib and
        # the undealt deck, which is the same approximation any player makes at
        # the table: you know what has appeared, not where the rest sits.
        unseen = list(info.unseen)
        pool = len(unseen)
        held = info.opp_hand_size

        best_card = None
        best_value = float("-inf")

        for card in legal:
            after = seq + [card]
            gain = score_play(after)
            count = info.count + CARD_VALUES[card]

            value = float(gain)
            if held and pool:
                room = MAX_COUNT - count
                replies = [c for c in unseen if CARD_VALUES[c] <= room]

                # Hypergeometric: the chance none of their cards can be played.
                blocked = pool - len(replies)
                p_stuck = (
                    comb(blocked, held) / comb(pool, held) if blocked >= held else 0.0
                )

                if replies:
                    mean_reply = sum(score_play(after + [r]) for r in replies) / len(replies)
                    value -= risk_weight * (1.0 - p_stuck) * mean_reply
                # If they cannot answer, this card very likely takes the go.
                value += p_stuck * 1.0

            # Ties go to the lower card, which keeps the count down and holds
            # the bigger cards back for later sub-rounds.
            if value > best_value or (
                value == best_value
                and best_card is not None
                and CARD_VALUES[card] < CARD_VALUES[best_card]
            ):
                best_value, best_card = value, card

        assert best_card is not None
        return best_card
