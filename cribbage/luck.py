"""Attributing a game's luck to the cut, per player, as a single number.

The starter is a *shared* card: when a five turns up, both players usually
benefit, so "I scored 12 off the cut" says very little on its own.  What matters
is the counterfactual -- given the hands that were actually kept, did the card
that came up favour you more than a typical card would have?

That question has an exact answer, because the counterfactual set is small and
fully known.  Once the discards are settled, both holdings, the crib and the
forty undealt cards are all fixed, and the only thing that could have gone
differently is which of those forty came up.  So for each round:

    luck = (what the real starter gave you) - (what an average starter gives you)

Subtracting the expectation does two useful things at once.  It removes the
dealer's systematic crib-and-heels edge, which is position rather than luck, and
it nets out cards that help both players, which contribute nothing to the
difference.  And because all forty candidates are enumerated rather than
sampled, the standard deviation comes out exactly too -- so "how many sd was
that" is available without any distributional assumption.

Two views are reported:

* **net** -- your show points minus your opponent's, against expectation.  This
  is the headline: it is exactly zero-sum, so one player's good cut luck is the
  other's bad luck by construction.
* **gross** -- your show points alone, against expectation.  Useful for "the
  deck ran hot for everyone" rounds, which net out to nothing.

Scope: this measures the cut *at the show*, which is where the starter's value
is realised -- hands, crib, his heels and nobs.  Pegging is left out; the
starter barely touches it and could not be attributed cleanly.  Rounds cut short
by someone winning before the show are excluded and counted separately.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Sequence

from .cards import JACK, rank_of
from .engine import CribbageState, CutContext, RoundRecord
from .scoring import score_hand

__all__ = [
    "RoundCutLuck", "CutLuck", "PercentileCut",
    "round_cut_luck", "cut_luck", "show_points",
]


def show_points(
    kept: Sequence[Sequence[int]],
    crib: Sequence[int],
    dealer: int,
    starter: int,
) -> tuple[float, float]:
    """Show points for (player 0, player 1) if ``starter`` were the cut.

    Hands, crib, his heels and nobs -- everything the starter is worth once the
    cards are down.
    """
    pone = 1 - dealer
    points = [0.0, 0.0]
    points[pone] = score_hand(kept[pone], starter)
    points[dealer] = score_hand(kept[dealer], starter)
    points[dealer] += score_hand(crib, starter, is_crib=True)
    if rank_of(starter) == JACK:
        points[dealer] += 2  # his heels
    return points[0], points[1]


def _show_points(record: RoundRecord, starter: int) -> tuple[float, float]:
    return show_points(record.kept, record.crib, record.dealer, starter)


@dataclass(frozen=True)
class RoundCutLuck:
    """How favourable one round's cut was, from ``player``'s side."""

    round_index: int
    player: int
    starter: int
    actual_net: float
    expected_net: float
    sd_net: float
    actual_gross: float
    expected_gross: float
    sd_gross: float

    @property
    def net(self) -> float:
        """Points the cut was worth to you *over your opponent*, net of expectation."""
        return self.actual_net - self.expected_net

    @property
    def gross(self) -> float:
        """Points the cut fed your own side, net of expectation."""
        return self.actual_gross - self.expected_gross

    @property
    def net_z(self) -> float:
        return self.net / self.sd_net if self.sd_net else 0.0


def round_cut_luck(record: RoundRecord, player: int) -> RoundCutLuck:
    """Score every card that could have been cut, and place the real one among them."""
    opponent = 1 - player

    nets: list[float] = []
    grosses: list[float] = []
    for candidate in record.candidates:
        points = _show_points(record, candidate)
        nets.append(points[player] - points[opponent])
        grosses.append(points[player])

    count = len(nets)
    mean_net = sum(nets) / count
    mean_gross = sum(grosses) / count
    var_net = sum((value - mean_net) ** 2 for value in nets) / count
    var_gross = sum((value - mean_gross) ** 2 for value in grosses) / count

    actual = _show_points(record, record.starter)
    return RoundCutLuck(
        round_index=record.round_index,
        player=player,
        starter=record.starter,
        actual_net=actual[player] - actual[opponent],
        expected_net=mean_net,
        sd_net=var_net ** 0.5,
        actual_gross=actual[player],
        expected_gross=mean_gross,
        sd_gross=var_gross ** 0.5,
    )


@dataclass(frozen=True)
class CutLuck:
    """A whole game's cut luck for one player."""

    player: int
    rounds: tuple[RoundCutLuck, ...]
    skipped: int = 0

    @property
    def net(self) -> float:
        """Total points the cuts were worth to you over your opponent."""
        return sum(row.net for row in self.rounds)

    @property
    def gross(self) -> float:
        return sum(row.gross for row in self.rounds)

    @property
    def net_sd(self) -> float:
        """Rounds are independent given the deals, so variances add."""
        return sum(row.sd_net ** 2 for row in self.rounds) ** 0.5

    @property
    def gross_sd(self) -> float:
        return sum(row.sd_gross ** 2 for row in self.rounds) ** 0.5

    @property
    def net_z(self) -> float:
        return self.net / self.net_sd if self.net_sd else 0.0

    @property
    def gross_z(self) -> float:
        return self.gross / self.gross_sd if self.gross_sd else 0.0

    def describe(self) -> str:
        """A one-line verdict, in the register someone would actually use."""
        z = self.net_z
        if z >= 2.0:
            verdict = "the cuts were a total banger"
        elif z >= 1.0:
            verdict = "the cuts ran clearly your way"
        elif z > -1.0:
            verdict = "the cuts were about par"
        elif z > -2.0:
            verdict = "the cuts ran against you"
        else:
            verdict = "the cuts were brutal"
        return (
            f"{self.net:+.1f} points from the cut over {len(self.rounds)} rounds "
            f"(sd {self.net_sd:.1f}, z = {z:+.2f}) -- {verdict}"
        )


def cut_luck(state: CribbageState, player: int) -> CutLuck:
    """How much the cuts favoured ``player`` over the whole game.

    Only rounds whose show actually ran are counted, since a round cut short by
    someone winning never realised its starter.  The number of skipped rounds is
    reported alongside.
    """
    completed = {event.round for event in state.events if event.kind == "crib"}
    rows = tuple(
        round_cut_luck(record, player)
        for record in state.round_records
        if record.round_index in completed
    )
    return CutLuck(
        player=player,
        rounds=rows,
        skipped=len(state.round_records) - len(rows),
    )


# ----------------------------------------------------------------------
# Dealing a cut of a chosen quality
# ----------------------------------------------------------------------


class PercentileCut:
    """Turn each round's starter at a fixed percentile of how good it could be.

    A seed cannot do this.  A seed has to be fixed before anyone discards, so it
    cannot know what the cut will be worth -- the same card is a gift beside one
    holding and a blank beside another.  The cut, though, happens *after* both
    discards, at which point the holdings and the crib are settled and every one
    of the forty remaining cards can be scored exactly.  So this ranks all forty
    by what they would be worth to ``player`` net of their opponent, and turns
    the one sitting at ``percentile``.

    100 is the best card in the deck for you every single round, 0 the worst, 50
    the median.  Note that a *per-round* percentile compounds over a game: nine
    rounds each at the 75th does not give a 75th-percentile game, it gives a far
    more lopsided one, since the edges add while the standard deviations only
    add in quadrature.
    """

    def __init__(self, player: int, percentile: float):
        if not 0 <= percentile <= 100:
            raise ValueError(f"percentile must be in [0, 100], got {percentile}")
        self.player = player
        self.percentile = percentile

    def __call__(self, context: CutContext) -> int:
        opponent = 1 - self.player
        scored = []
        for card in context.deck:
            points = show_points(context.kept, context.crib, context.dealer, card)
            # Sort by (value, card) so ties resolve deterministically.
            scored.append((points[self.player] - points[opponent], card))
        scored.sort()
        position = round((self.percentile / 100) * (len(scored) - 1))
        return scored[position][1]

    def __repr__(self) -> str:
        return f"PercentileCut(player={self.player}, percentile={self.percentile})"
