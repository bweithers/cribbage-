"""Position-aware play: the same expected-value machinery, aimed at winning.

:class:`~cribbage.agents.heuristic.HeuristicAgent` maximizes points and never
reads the score.  Cribbage is a race to 121, not a points-accumulation game, so
that is wrong -- the question this module exists to answer is *how* wrong.

Two mechanisms, deliberately different in character:

**Stance** is continuous and always active.  A player ahead on the board should
play safe: deny the opponent's crib, refuse pegging risk, and prefer a keep that
scores reliably over one that might score big.  A player behind should do the
opposite and reach for variance.  ``stance`` runs from +1 (comfortably ahead,
defend) to -1 (behind, attack), and drives three separate channels so each can
be ablated on its own.

**Endgame** is sharp and rare.  When going out this deal is actually in reach,
expected points stop mattering and only the probability of getting there does.
That probability is free: the discard already enumerates all 46 starters, so the
exact score distribution is in hand, not just its mean.

The par figures below are measured from this simulator rather than copied from a
book -- regenerate them with ``scripts/measure_par.py``.  They agree with the
published averages (a dealer scores about 16 points a deal and a pone about 10),
which is a small extra check on the engine.
"""

from __future__ import annotations

import math
from typing import Sequence

from ..cards import rank_of
from ..engine import InfoState
from ..scoring import score_play
from .heuristic import DiscardOption, HeuristicAgent

__all__ = ["PositionalAgent", "PAR_DEALER", "PAR_PONE"]

#: Mean points a player scores in a deal they dealt, and one they did not.
PAR_DEALER = 16.1
PAR_PONE = 10.1

#: Non-hand points expected in a deal: pegging, gos and his heels.
PEG_DEALER = 3.4
PEG_PONE = 2.0


def _normalized(counts: dict[int, float]) -> tuple[tuple[int, float], ...]:
    total = sum(counts.values())
    return tuple((points, weight / total) for points, weight in sorted(counts.items()))


#: Distribution of those non-hand points, measured by scripts/measure_par.py.
#: The mean alone is not enough for an endgame calculation: pegging has a
#: standard deviation of about two points, which is the same order as the spread
#: of a hand's score across starters.  Thresholding on the mean therefore aims at
#: a target that is itself uncertain by more than the thing being optimized.
PEG_DISTRIBUTION_DEALER = _normalized({
    1: 0.213, 2: 0.181, 3: 0.205, 4: 0.157, 5: 0.098,
    6: 0.062, 7: 0.035, 8: 0.019, 9: 0.013,
})
PEG_DISTRIBUTION_PONE = _normalized({
    0: 0.195, 1: 0.334, 2: 0.154, 3: 0.160,
    4: 0.057, 5: 0.025, 6: 0.034, 7: 0.025,
})

#: Rough win-probability value of one point of margin, used to put expected
#: points and "I go out this deal" on the same scale.  A ten point lead is worth
#: something like ten to fifteen points of win probability.
POINT_VALUE = 0.012

#: Lead, in points, at which stance is most of the way to fully committed.
LEAD_SCALE = 18.0

FIVE = 4  # rank index of a five


class PositionalAgent(HeuristicAgent):
    """Heuristic play plus awareness of the score.

    Args:
        endgame: when going out this deal is reachable, maximize the probability
            of doing so instead of expected points.  Self-regulating: when every
            lay-away is certain to go out, or none can, the probability term
            stops discriminating and expected points decides again.
        endgame_smoothing: fold the going-out probability over the measured
            distribution of pegging points rather than thresholding on its mean.
            Without this the objective is a sharp cut at a threshold that is
            itself uncertain by about two points.
        stance_variance: prefer reliable keeps when ahead and swingy ones when
            behind, scaled by the spread of the keep's score across starters.
        stance_crib: defend the opponent's crib harder when ahead.
        stance_peg: fear the opponent's pegging reply more when ahead.
        crib_defense: constant extra weight on the opponent's crib, independent
            of the score.  The "never throw points into their crib" knob.
        five_penalty: extra points of penalty for laying a five into the
            opponent's crib, over and above what the crib table already says it
            costs.  Tests whether the folk rule adds anything to the arithmetic.
        finish: play a card that reaches the target immediately, whatever else
            the reply analysis thinks.
    """

    name = "positional"

    def __init__(
        self,
        endgame: bool = True,
        endgame_smoothing: bool = True,
        stance_variance: float = 0.0,
        stance_crib: float = 0.0,
        stance_peg: float = 0.0,
        crib_defense: float = 0.0,
        five_penalty: float = 0.0,
        finish: bool = True,
        lead_scale: float = LEAD_SCALE,
        point_value: float = POINT_VALUE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.endgame = endgame
        self.endgame_smoothing = endgame_smoothing
        self.stance_variance = stance_variance
        self.stance_crib = stance_crib
        self.stance_peg = stance_peg
        self.crib_defense = crib_defense
        self.five_penalty = five_penalty
        self.finish = finish
        self.lead_scale = lead_scale
        self.point_value = point_value

    # ------------------------------------------------------------------
    # Position
    # ------------------------------------------------------------------

    def stance(self, info: InfoState) -> float:
        """+1 comfortably ahead and playing safe, -1 behind and reaching.

        Based on the lead rather than on absolute score, because what decides
        whether to gamble is the gap, not how far along the board you are.  The
        deal is worth about six points of edge, so holding it counts as being
        that much further ahead.
        """
        lead = info.my_score - info.opp_score
        lead += (PAR_DEALER - PAR_PONE) / 2 * (1 if info.is_dealer else -1)
        return math.tanh(lead / self.lead_scale)

    # ------------------------------------------------------------------
    # Discard
    # ------------------------------------------------------------------

    def crib_weight_for(self, info: InfoState) -> float:
        """Weight on the crib term.  Only the opponent's crib gets defended."""
        if info.is_dealer:
            return self.crib_weight
        extra = self.crib_defense + self.stance_crib * max(0.0, self.stance(info))
        return self.crib_weight + extra

    def points_needed_from_hand(self, option: DiscardOption, info: InfoState) -> float:
        """Points the hand must produce for this deal to finish the game.

        Subtracts what the rest of the deal is expected to bring in: pegging
        either way, plus this lay-away's crib when dealing.  Expected rather than
        exact, because pegging is not knowable at discard time.
        """
        to_go = info.target - info.my_score
        if info.is_dealer:
            return to_go - PEG_DEALER - option.crib
        return to_go - PEG_PONE

    def probability_of_going_out(
        self, option: DiscardOption, info: InfoState
    ) -> float:
        """Chance this lay-away finishes the game this deal.

        The hand's contribution is known exactly -- every starter has been
        enumerated -- so the only uncertainty left is how much pegging will add.
        Folding over the measured pegging distribution turns a sharp threshold
        into a weighted sum, which stops single-starter noise from swamping real
        differences in expected points.
        """
        if not self.endgame_smoothing:
            return option.probability_at_least(
                self.points_needed_from_hand(option, info)
            )

        to_go = info.target - info.my_score
        crib = option.crib if info.is_dealer else 0.0
        distribution = (
            PEG_DISTRIBUTION_DEALER if info.is_dealer else PEG_DISTRIBUTION_PONE
        )
        return sum(
            weight * option.probability_at_least(to_go - pegged - crib)
            for pegged, weight in distribution
        )

    def discard_value(self, option: DiscardOption, info: InfoState) -> float:
        crib = option.crib * self.crib_weight_for(info)
        points = option.mean + (crib if info.is_dealer else -crib)

        if self.five_penalty and not info.is_dealer:
            points -= self.five_penalty * sum(
                1 for card in option.pair if rank_of(card) == FIVE
            )

        if self.stance_variance:
            # Ahead: shave the spread, take the reliable hand.  Behind: add it.
            points -= self.stance_variance * self.stance(info) * option.stdev

        if not self.endgame:
            return points

        # Both terms are win probability: going out this deal wins outright, and
        # points left over are worth roughly a percent each.
        return self.probability_of_going_out(option, info) + self.point_value * points

    # ------------------------------------------------------------------
    # Play
    # ------------------------------------------------------------------

    def effective_risk(self, info: InfoState) -> float:
        """Fear the opponent's reply more when protecting a lead."""
        if not self.stance_peg:
            return self.risk_weight
        scaled = self.risk_weight * (1.0 + self.stance_peg * self.stance(info))
        return max(0.0, min(3.0, scaled))

    def play(self, info: InfoState) -> int:
        if self.finish:
            to_go = info.target - info.my_score
            seq = list(info.seq)
            for card in info.legal:
                if score_play(seq + [card]) >= to_go:
                    return card
        return super().play(info)

    def discard(self, info: InfoState) -> Sequence[int]:
        return super().discard(info)
