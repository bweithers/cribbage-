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

from itertools import groupby
from math import comb
from typing import Sequence

from ..cards import CARD_VALUES, rank_of
from ..engine import MAX_COUNT, InfoState
from ..peg_table import peg_ev
from ..scoring import score_play
from .heuristic import DiscardOption, HeuristicAgent

__all__ = ["PeggingAwareAgent", "expected_max"]


def expected_max(
    values: Sequence[float], unplayable: int, held: int, pool: int
) -> tuple[float, float]:
    """``E[max]`` over the ``held`` cards an opponent drew from ``pool``.

    ``values`` are the outcomes of the cards they could legally play; the other
    ``unplayable`` cards in the pool are cards that will not fit under 31.  If
    every card they hold is one of those they cannot answer at all, which is the
    second return value.

    Exact rather than sampled: the distribution of the maximum over a
    hypergeometric draw is just ``C(cards at or below v, held) / C(pool, held)``,
    and the pool is forty cards, so it can be summed over directly.
    """
    total = comb(pool, held)
    if not total or held <= 0:
        return 0.0, 0.0

    stuck = comb(unplayable, held) / total if unplayable >= held else 0.0
    expected = 0.0
    previous = stuck
    seen = unplayable
    for value, group in groupby(sorted(values)):
        seen += sum(1 for _ in group)
        reached = comb(seen, held) / total if seen >= held else 0.0
        expected += value * (reached - previous)
        previous = reached
    return expected, stuck


class PeggingAwareAgent(HeuristicAgent):
    """Expected hand and crib, plus the expected pegging value of the keep."""

    name = "pegging"

    def __init__(self, peg_weight: float = 0.5, best_reply: bool = False,
                 play_depth: int = 1, **kwargs):
        super().__init__(**kwargs)
        #: Scales the pegging term in the discard.  0.0 drops it entirely.
        #: Defaults below 1.0 deliberately: this term is *sampled* where the
        #: hand and crib terms are exact, and a noisier estimate deserves to be
        #: trusted less.  The measured optimum sits near a half.
        self.peg_weight = peg_weight
        #: Assume the opponent finds their best reply rather than an average
        #: one.  More principled, and **measured null** -- see the class docs.
        self.best_reply = best_reply
        #: 1 looks at the opponent's reply; 2 also at my answer to it.
        self.play_depth = play_depth

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

        values = []
        unplayable = 0
        for card in unseen:
            if CARD_VALUES[card] <= room:
                values.append(float(score_play(list(after) + [card])))
            else:
                unplayable += 1
        return expected_max(values, unplayable, held, pool)

    def play(self, info: InfoState) -> int:
        """One-ply by default; ``play_depth=2`` searches a move further.

        At depth two, each of my cards is judged by what the opponent's *best*
        answer to it nets them once my own best follow-up is subtracted -- so a
        card that hands them a pair is forgiven if it sets up a run for me.
        """
        if self.play_depth < 2:
            return super().play(info)

        legal = list(info.legal)
        if len(legal) == 1:
            return legal[0]

        risk = self.effective_risk(info)
        seq = list(info.seq)
        unseen = list(info.unseen)
        pool = len(unseen)
        held = info.opp_hand_size

        best_card = None
        best_value = float("-inf")
        for card in legal:
            after = seq + [card]
            value = float(score_play(after))
            count = info.count + CARD_VALUES[card]
            mine_left = [c for c in info.hand if c != card]

            if held and pool:
                room = MAX_COUNT - count
                nets = []
                unplayable = 0
                for reply in unseen:
                    if CARD_VALUES[reply] > room:
                        unplayable += 1
                        continue
                    theirs = score_play(after + [reply])
                    room_after = MAX_COUNT - count - CARD_VALUES[reply]
                    mine = max(
                        (score_play(after + [reply, follow])
                         for follow in mine_left
                         if CARD_VALUES[follow] <= room_after),
                        default=0,
                    )
                    nets.append(float(theirs) - mine)
                reply_cost, stuck = expected_max(nets, unplayable, held, pool)
                value -= risk * reply_cost
                value += stuck * 1.0

            if value > best_value or (
                value == best_value and best_card is not None
                and CARD_VALUES[card] < CARD_VALUES[best_card]
            ):
                best_value, best_card = value, card

        assert best_card is not None
        return best_card

    def pegging_value(self, option: DiscardOption, info: InfoState) -> float:
        """Expected pegging points for this keep, net of the opponent's."""
        return peg_ev([rank_of(card) for card in option.keep], info.is_dealer)

    def discard_value(self, option: DiscardOption, info: InfoState) -> float:
        base = super().discard_value(option, info)
        if not self.peg_weight:
            return base
        return base + self.peg_weight * self.pegging_value(option, info)
