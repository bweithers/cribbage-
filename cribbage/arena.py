"""Running matches and reporting results honestly.

Two things here are less obvious than they look:

* **Alternating the first deal.**  The dealer holds a real edge -- the crib.
  Fixing agent A in seat 0 and always dealing to seat 0 first would hand A that
  edge in every game.  Games alternate which seat deals first, so the seats are
  symmetric over an even number of games.
* **Reporting a confidence interval.**  A win rate quoted without one invites
  reading noise as a result.  Every win rate here comes with a Wilson score
  interval, so a 200-game "51%" is visibly indistinguishable from a coin.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from .agents.base import Agent
from .engine import DEFAULT_TARGET, POINT_KINDS, CribbageState, GameResult

__all__ = [
    "play_game", "run_match", "run_paired_match",
    "MatchStats", "PairedResult", "wilson_interval",
]


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- well behaved near 0 and 1, unlike the normal one."""
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    spread = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    half = spread / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def play_game(
    agents: Sequence[Agent],
    seed: Optional[int] = None,
    dealer: int = 0,
    target: int = DEFAULT_TARGET,
    cut_seed: Optional[int] = None,
) -> CribbageState:
    """Play one game to completion and return the finished state.

    Agents are handed an :class:`~cribbage.engine.InfoState` and nothing else,
    so an agent cannot reach the hidden state even by accident.
    """
    state = CribbageState(dealer=dealer, target=target, seed=seed, cut_seed=cut_seed)
    for agent in agents:
        agent.reset()

    while not state.is_terminal():
        player = state.current_player
        assert player is not None
        action = agents[player].act(state.information_state(player))
        state.apply_action(action)

    return state


@dataclass
class MatchStats:
    """Aggregated results.  Index 0/1 throughout refers to seat, not to agent order."""

    names: tuple[str, str]
    games: int = 0
    wins: list[int] = field(default_factory=lambda: [0, 0])
    skunks: list[int] = field(default_factory=lambda: [0, 0])
    double_skunks: list[int] = field(default_factory=lambda: [0, 0])
    total_score: list[int] = field(default_factory=lambda: [0, 0])
    total_margin: int = 0
    total_rounds: int = 0
    points: list[dict[str, int]] = field(
        default_factory=lambda: [defaultdict(int), defaultdict(int)]
    )

    def record(self, state: CribbageState) -> None:
        result: GameResult = state.result()
        self.games += 1
        self.wins[result.winner] += 1
        if result.skunk:
            self.skunks[result.winner] += 1
        if result.double_skunk:
            self.double_skunks[result.winner] += 1
        self.total_score[0] += result.scores[0]
        self.total_score[1] += result.scores[1]
        self.total_margin += result.margin
        self.total_rounds += result.rounds
        for event in state.events:
            if event.points:
                self.points[event.player][event.kind] += event.points

    @property
    def win_rate(self) -> float:
        return self.wins[0] / self.games if self.games else 0.0

    def points_per_round(self, seat: int, kind: str) -> float:
        rounds = self.total_rounds
        return self.points[seat].get(kind, 0) / rounds if rounds else 0.0

    def report(self) -> str:
        if not self.games:
            return "no games played"

        low, high = wilson_interval(self.wins[0], self.games)
        width = max(len(self.names[0]), len(self.names[1]), 10)
        lines = [
            f"{self.names[0]}  vs  {self.names[1]}      {self.games} games",
            "",
            f"  {self.names[0]:<{width}} wins {self.wins[0]:>6}  "
            f"({self.win_rate:6.1%}, 95% CI {low:.1%}-{high:.1%})",
            f"  {self.names[1]:<{width}} wins {self.wins[1]:>6}  "
            f"({1 - self.win_rate:6.1%})",
            "",
            f"  mean final score      {self.total_score[0] / self.games:6.1f}"
            f"  vs {self.total_score[1] / self.games:6.1f}",
            f"  mean winning margin   {self.total_margin / self.games:6.1f}",
            f"  mean rounds per game  {self.total_rounds / self.games:6.1f}",
            "",
            f"  skunks                {self.skunks[0]:>6}  vs {self.skunks[1]:>6}"
            f"   (double: {self.double_skunks[0]} vs {self.double_skunks[1]})",
            "",
            "  points per round      "
            f"{self.names[0][:12]:>12}  {self.names[1][:12]:>12}",
        ]
        labels = {
            "hand": "hand",
            "crib": "crib",
            "play": "pegging",
            "go": "go / last card",
            "heels": "his heels",
        }
        for kind in POINT_KINDS:
            lines.append(
                f"    {labels[kind]:<18}"
                f"{self.points_per_round(0, kind):>12.2f}"
                f"  {self.points_per_round(1, kind):>12.2f}"
            )
        totals = [
            sum(self.points[seat].values()) / self.total_rounds for seat in (0, 1)
        ]
        lines.append(f"    {'total':<18}{totals[0]:>12.2f}  {totals[1]:>12.2f}")
        return "\n".join(lines)


def run_match(
    agents: Sequence[Agent],
    games: int,
    seed: int = 0,
    target: int = DEFAULT_TARGET,
    progress: Optional[Callable[[int, int], None]] = None,
) -> MatchStats:
    """Play ``games`` games, alternating who deals first."""
    stats = MatchStats(names=(agents[0].name, agents[1].name))
    rng = random.Random(seed)
    for index in range(games):
        state = play_game(
            agents,
            seed=rng.randrange(1 << 30),
            dealer=index % 2,  # alternate the deal so neither seat keeps the edge
            target=target,
        )
        stats.record(state)
        if progress is not None:
            progress(index + 1, games)
    return stats


# ----------------------------------------------------------------------
# Paired comparison
# ----------------------------------------------------------------------


@dataclass
class PairedResult:
    """Outcome of a mirrored comparison between two agents.

    Each pair is the *same deal sequence* played twice with the seats swapped,
    so both agents get to hold both sides of every hand.  That cancels most of
    the deal luck, which in cribbage is large enough to drown small differences
    in play.

    The confidence interval is computed on the per-pair scores, not on the
    individual games.  Two games from one pair are strongly correlated, so
    treating them as independent trials would understate the interval.
    """

    name_a: str
    name_b: str
    pairs: int = 0
    wins_a: int = 0
    swept_a: int = 0        # pairs where A won both sides of the same deal
    swept_b: int = 0
    split: int = 0
    _scores: list[float] = field(default_factory=list)

    def record_pair(self, a_won_first: bool, a_won_second: bool) -> None:
        wins = int(a_won_first) + int(a_won_second)
        self.pairs += 1
        self.wins_a += wins
        self._scores.append(wins / 2)
        if wins == 2:
            self.swept_a += 1
        elif wins == 0:
            self.swept_b += 1
        else:
            self.split += 1

    def merge(self, other: "PairedResult") -> None:
        self.pairs += other.pairs
        self.wins_a += other.wins_a
        self.swept_a += other.swept_a
        self.swept_b += other.swept_b
        self.split += other.split
        self._scores.extend(other._scores)

    @property
    def games(self) -> int:
        return self.pairs * 2

    @property
    def win_rate(self) -> float:
        return self.wins_a / self.games if self.games else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        """95% interval for A's win rate, from the per-pair scores."""
        n = len(self._scores)
        if n < 2:
            return (0.0, 1.0)
        mean = sum(self._scores) / n
        variance = sum((x - mean) ** 2 for x in self._scores) / (n - 1)
        half = 1.96 * math.sqrt(variance / n)
        return (max(0.0, mean - half), min(1.0, mean + half))

    @property
    def significant(self) -> bool:
        low, high = self.interval
        return low > 0.5 or high < 0.5

    def summary(self) -> str:
        low, high = self.interval
        verdict = "" if self.significant else "  (not significant)"
        return (
            f"{self.name_a} vs {self.name_b}: {self.win_rate:6.2%} "
            f"[{low:.2%}, {high:.2%}] over {self.games} games{verdict}"
        )


def run_paired_match(
    factory_a: Callable[[], Agent],
    factory_b: Callable[[], Agent],
    pairs: int,
    seed: int = 0,
    target: int = DEFAULT_TARGET,
    name_a: Optional[str] = None,
    name_b: Optional[str] = None,
) -> PairedResult:
    """Play each deal sequence twice, once from each seat, and score agent A.

    Both games of a pair use the same engine seed, so they deal identical cards;
    only which agent holds which side changes.

    Agents are rebuilt from their factories for every game rather than reused.
    For a deterministic agent that changes nothing and keeps the two halves of a
    pair as symmetric as possible, which is the whole point.  A *stochastic*
    agent built from a fixed-seed factory will therefore replay the same random
    sequence in every game; if that is not what you want, hand in a factory that
    varies the seed itself.
    """
    probe_a, probe_b = factory_a(), factory_b()
    result = PairedResult(
        name_a=name_a or probe_a.name, name_b=name_b or probe_b.name
    )
    rng = random.Random(seed)

    for _ in range(pairs):
        game_seed = rng.randrange(1 << 30)
        first = play_game([factory_a(), factory_b()], seed=game_seed, dealer=0, target=target)
        second = play_game([factory_b(), factory_a()], seed=game_seed, dealer=0, target=target)
        result.record_pair(first.result().winner == 0, second.result().winner == 1)

    return result
