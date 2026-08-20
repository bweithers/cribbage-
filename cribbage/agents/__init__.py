"""Baseline agents, and the registry the CLI resolves names through."""

from __future__ import annotations

from typing import Callable

from .base import Agent
from .heuristic import HeuristicAgent
from .positional import PositionalAgent
from .random_agent import RandomAgent

__all__ = [
    "Agent", "RandomAgent", "HeuristicAgent", "PositionalAgent",
    "AGENTS", "make_agent",
]

#: Name -> factory.  ``make_agent`` passes a seed to those that accept one.
AGENTS: dict[str, Callable[..., Agent]] = {
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
    "greedy": lambda **kw: HeuristicAgent(risk_weight=0.0),
    # Measured at parity with `heuristic` -- see the experiment section of the
    # README.  Registered so the CLI and the ablation harness can reach it, not
    # because it is stronger.  `endgame_smoothing` stays on: the unsmoothed
    # objective measured significantly worse.
    "positional": lambda **kw: PositionalAgent(
        stance_variance=0.3, stance_crib=0.4, stance_peg=0.5
    ),
}


def make_agent(name: str, seed: int | None = None) -> Agent:
    """Build an agent by name, seeding it when it takes a seed."""
    try:
        factory = AGENTS[name]
    except KeyError:
        known = ", ".join(sorted(AGENTS))
        raise ValueError(f"unknown agent {name!r}; known agents: {known}") from None
    try:
        agent = factory(seed=seed)
    except TypeError:
        agent = factory()
    # Report under the registry key, so variants of one class stay distinguishable.
    agent.name = name
    return agent
