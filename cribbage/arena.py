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

__all__ = ["play_game", "run_match", "MatchStats", "wilson_interval"]


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
) -> CribbageState:
    """Play one game to completion and return the finished state.

    Agents are handed an :class:`~cribbage.engine.InfoState` and nothing else,
    so an agent cannot reach the hidden state even by accident.
    """
    state = CribbageState(dealer=dealer, target=target, seed=seed)
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
