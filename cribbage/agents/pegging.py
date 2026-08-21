"""A discard policy that also cares how the cards will play.

:class:`~cribbage.agents.heuristic.HeuristicAgent` chooses a keep by maximizing
what it will *show* -- hand score plus or minus the crib -- and is completely
blind to what happens in between.  That is its largest known gap, and unlike the
position features (which measured null because they perturb only a couple of
percent of discards) this one distorts *every* hand: four fives are worth twenty
at the show and peg abominably, A-2-3-4 shows six and pegs beautifully, and
nothing in an expected-hand-score calculation can tell them apart.

The fix is one extra term, from :mod:`cribbage.peg_table`::

    value = E[hand] ± E[crib] + peg_weight × E[pegging − their pegging]

All three are in points, so 1.0 is the weight the units argue for.  Measured, the
best weight is about **0.5**, which makes sense: the pegging term is *sampled*
where the hand and crib terms are exact, and shrinking a noisy estimate toward
zero is the right response to that noise.  At 0.5 this wins 50.83% against the
baseline over ten thousand mirrored games, replicated on fresh seeds -- the first
thing in this repository to beat it.

There is a second, tempting idea here that **does not work**, kept behind
``best_reply`` so the negative result is not lost.  The baseline's play policy
averages over the cards the opponent might hold, which quietly assumes they pick
one at random; assuming instead that they find their *best* card is obviously
more correct, and gives far more realistic numbers (leading a four against four
unknown cards costs 0.62 expected points rather than 0.17).  It also measures
49.85%, dead null, because it changes barely half a percent of play decisions --
scaling every risk up together mostly preserves their ranking.  Being right about
the opponent's model is not the same as making different decisions.
"""

from __future__ import annotations

from math import comb
from typing import Sequence

from ..cards import CARD_VALUES, rank_of
from ..engine import InfoState
from ..peg_table import peg_ev
from ..scoring import score_play
from .heuristic import DiscardOption, HeuristicAgent

__all__ = ["PeggingAwareAgent"]


class PeggingAwareAgent(HeuristicAgent):
    """Expected hand and crib, plus the expected pegging value of the keep."""

    name = "pegging"

    def __init__(self, peg_weight: float = 0.5, best_reply: bool = False, **kwargs):
        super().__init__(**kwargs)
        #: Scales the pegging term in the discard.  0.0 drops it entirely.
        #: Defaults below 1.0 deliberately: this term is *sampled* where the
        #: hand and crib terms are exact, and a noisier estimate deserves to be
        #: trusted less.  The measured optimum sits near a half.
        self.peg_weight = peg_weight
        #: Assume the opponent finds their best reply rather than an average
        #: one.  More principled, and **measured null** -- see the class docs.
        self.best_reply = best_reply

    def reply_cost(
        self, after: list[int], unseen: Sequence[int], room: int,
        held: int, pool: int,
    ) -> tuple[float, float]:
        """Expected reply points assuming the opponent plays their *best* card.

        The baseline averages over every card the opponent might hold, which
        quietly assumes they pick one at random.  They do not: they hold ``held``
        cards drawn from ``pool`` and will play whichever scores most.  So take
        the expectation of the *maximum* instead, which is exact here -- the
        distribution of that maximum is hypergeometric and the pool is small
        enough to sum over directly.
        """
        if not self.best_reply:
            return super().reply_cost(after, unseen, room, held, pool)

        total = comb(pool, held)
        if not total:
            return 0.0, 0.0

        # -1 marks a card that cannot legally be played; the opponent's outcome
        # is the maximum over the cards they hold, and a maximum of -1 is a go.
        values = sorted(
            score_play(list(after) + [card]) if CARD_VALUES[card] <= room else -1
            for card in unseen
        )

        expected = 0.0
        stuck = 0.0
        previous = 0.0
        index = 0
        while index < len(values):
            value = values[index]
            while index < len(values) and values[index] == value:
                index += 1
            reached = comb(index, held) / total if index >= held else 0.0
            probability = reached - previous
            previous = reached
            if value < 0:
                stuck = reached
            elif value > 0:
                expected += value * probability
        return expected, stuck

    def pegging_value(self, option: DiscardOption, info: InfoState) -> float:
        """Expected pegging points for this keep, net of the opponent's."""
        return peg_ev([rank_of(card) for card in option.keep], info.is_dealer)

    def discard_value(self, option: DiscardOption, info: InfoState) -> float:
        base = super().discard_value(option, info)
        if not self.peg_weight:
            return base
        return base + self.peg_weight * self.pegging_value(option, info)
