#!/usr/bin/env python3
"""Measure the per-deal constants that cribbage/agents/positional.py relies on.

The positional agent needs to know what a normal deal is worth in order to judge
whether a player is close to going out.  Those numbers are measured from this
simulator rather than copied from a book, so this script exists to regenerate
them when anything about the agents or the rules changes.

It also reports the distribution of non-hand points (pegging, gos, his heels),
which is what an endgame calculation has to fold over: the hand's score is known
exactly across all 46 starters, but how much pegging will add is not.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cribbage.agents import make_agent  # noqa: E402
from cribbage.arena import play_game  # noqa: E402

HAND_KINDS = ("hand", "crib")
PEG_KINDS = ("play", "go", "heels")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--games", type=int, default=600)
    parser.add_argument("--agent", default="heuristic")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    totals = {True: [], False: []}          # keyed by "dealt this hand"
    peg_totals = {True: [], False: []}
    by_kind = {True: defaultdict(int), False: defaultdict(int)}
    peg_hist = {True: Counter(), False: Counter()}

    rng = random.Random(args.seed)
    for game in range(args.games):
        agents = [make_agent(args.agent), make_agent(args.agent)]
        state = play_game(agents, seed=rng.randrange(1 << 30), dealer=game % 2)

        first_dealer = state.events[0].player
        last_round = state.round_index
        per_round = defaultdict(lambda: [0, 0])
        per_round_peg = defaultdict(lambda: [0, 0])
        for event in state.events:
            if not event.points or event.round == last_round:
                continue  # the final round is cut short by the win
            per_round[event.round][event.player] += event.points
            if event.kind in PEG_KINDS:
                per_round_peg[event.round][event.player] += event.points
            dealt = event.player == (first_dealer + event.round - 1) % 2
            by_kind[dealt][event.kind] += event.points

        for round_index, points in per_round.items():
            dealer = (first_dealer + round_index - 1) % 2
            totals[True].append(points[dealer])
            totals[False].append(points[1 - dealer])
            peg = per_round_peg[round_index]
            peg_totals[True].append(peg[dealer])
            peg_totals[False].append(peg[1 - dealer])
            peg_hist[True][peg[dealer]] += 1
            peg_hist[False][peg[1 - dealer]] += 1

    deals = len(totals[True])
    print(f"{args.agent} vs {args.agent}: {args.games} games, {deals} completed deals\n")

    print("  points per deal")
    for dealt, label in ((True, "PAR_DEALER"), (False, "PAR_PONE")):
        values = totals[dealt]
        print(
            f"    {label:<12} {statistics.mean(values):6.2f}"
            f"   sd {statistics.pstdev(values):5.2f}"
            f"   median {statistics.median(values):5.1f}"
            f"   p90 {sorted(values)[int(len(values) * 0.9)]:3d}"
            f"   max {max(values):3d}"
        )

    print("\n  non-hand points per deal (pegging, gos, his heels)")
    for dealt, label in ((True, "PEG_DEALER"), (False, "PEG_PONE")):
        values = peg_totals[dealt]
        print(
            f"    {label:<12} {statistics.mean(values):6.2f}"
            f"   sd {statistics.pstdev(values):5.2f}"
        )

    print("\n  breakdown per deal")
    for kind in HAND_KINDS + PEG_KINDS:
        print(
            f"    {kind:<8} dealer {by_kind[True][kind] / deals:5.2f}"
            f"   pone {by_kind[False][kind] / deals:5.2f}"
        )

    print("\n  pegging distribution (what an endgame calculation must fold over)")
    for dealt, label in ((True, "dealer"), (False, "pone")):
        hist = peg_hist[dealt]
        total = sum(hist.values())
        row = "  ".join(
            f"{points}:{hist[points] / total:.3f}"
            for points in sorted(hist)
            if hist[points] / total >= 0.01
        )
        print(f"    {label:<7} {row}")

    print("\n  paste into cribbage/agents/positional.py:")
    print(f"    PAR_DEALER = {statistics.mean(totals[True]):.1f}")
    print(f"    PAR_PONE = {statistics.mean(totals[False]):.1f}")
    print(f"    PEG_DEALER = {statistics.mean(peg_totals[True]):.1f}")
    print(f"    PEG_PONE = {statistics.mean(peg_totals[False]):.1f}")
    print("\n  and the pegging distributions, as {points: probability} dicts:")
    for dealt, label in ((True, "PEG_DISTRIBUTION_DEALER"), (False, "PEG_DISTRIBUTION_PONE")):
        hist = peg_hist[dealt]
        total = sum(hist.values())
        body = ", ".join(
            f"{points}: {hist[points] / total:.3f}"
            for points in sorted(hist)
            if hist[points] / total >= 0.01
        )
        print(f"    {label} = _normalized({{{body}}})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
