#!/usr/bin/env python3
"""How much of deep play search is locked behind knowing the opponent's hand?

Solving the pegging subgame exactly is easy once both hands are known -- eight
plies, four choices a node.  The hard part is that they are not known.  This
separates the two by measuring three policies against the same opponent:

    heuristic      one ply against a marginal over the unseen cards
    PIMC           the subgame solved exactly against *sampled* holdings
    cheating       the subgame solved exactly against the *actual* holding

The cheating policy is not a legal agent -- it reads the opponent's hand off the
engine -- which is the point.  It is the ceiling that exact search would reach if
the inference problem were solved, and the gap between it and PIMC is the price
of guessing.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cribbage.agents import (  # noqa: E402
    HeuristicAgent, PeggingAwareAgent, PimcPeggingAgent,
)
from cribbage.engine import CribbageState, Phase  # noqa: E402
from cribbage.pegsolve import best_cards  # noqa: E402


def cheating_play(state: CribbageState) -> int:
    """Solve against the opponent's real hand.  Illegal; that is the point."""
    player = state.current_player
    assert player is not None
    last = None if state.last_to_play is None else (
        0 if state.last_to_play == player else 1
    )
    values = best_cards(
        state.hands[player], state.hands[1 - player],
        count=state.count, seq=state.seq, last=last,
    )
    return max(values, key=lambda card: (values[card], -card))


def run(policy, games: int, seed: int, samples: int):
    """Seat 0 pegs by ``policy``; seat 1 is always the plain heuristic."""
    opponent = HeuristicAgent()
    discarder = PeggingAwareAgent(peg_weight=0.5)
    pimc = PimcPeggingAgent(peg_weight=0.5, samples=samples)
    rng = random.Random(seed)

    per_round: list[float] = []
    wins = 0
    for game in range(games):
        state = CribbageState(dealer=game % 2, seed=rng.randrange(1 << 30))
        while not state.is_terminal():
            player = state.current_player
            info = state.information_state(player)
            if info.phase is Phase.DISCARD:
                state.apply_action(
                    (discarder if player == 0 else opponent).act(info)
                )
            elif player != 0:
                state.apply_action(opponent.play(info))
            elif policy == "cheating":
                state.apply_action(cheating_play(state))
            elif policy == "pimc":
                state.apply_action(pimc.play(info))
            else:
                state.apply_action(opponent.play(info))

        pegged = [0, 0]
        for event in state.events:
            if event.kind in ("play", "go"):
                pegged[event.player] += event.points
        per_round.append((pegged[0] - pegged[1]) / state.round_index)
        wins += state.result().winner == 0
    return statistics.mean(per_round), statistics.pstdev(per_round), wins / games


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--games", type=int, default=400)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    print(f"{args.games} games per row, seat 1 always the plain heuristic.\n")
    print(f"  {'seat 0 pegs by':<24}{'net peg/round':>15}{'95% CI':>18}"
          f"{'win rate':>10}")
    print("  " + "-" * 67)

    for label, policy in (
        ("heuristic (control)", "heuristic"),
        (f"PIMC, {args.samples} sampled hands", "pimc"),
        ("exact solve, CHEATING", "cheating"),
    ):
        mean, sd, wins = run(policy, args.games, args.seed, args.samples)
        half = 1.96 * sd / (args.games ** 0.5)
        print(f"  {label:<24}{mean:>+15.3f}"
              f"{f'[{mean - half:+.3f}, {mean + half:+.3f}]':>18}{wins:>10.1%}")

    print(
        "\n  The gap between the last two rows is what knowing the opponent's\n"
        "  hand is worth. Depth alone does not collect it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
