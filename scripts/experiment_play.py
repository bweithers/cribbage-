#!/usr/bin/env python3
"""What is the pegging *play* actually worth, and does looking deeper help?

Built to answer a claim that turned out to be wrong.  The best-reply opponent
model measured null, and that got over-generalised into "nothing improves the
play" -- when an earlier run had already shown the one-ply reply term beating a
greedy policy handily.

So this measures the ladder properly.  Each rung differs from the baseline in
exactly one respect, so each match measures one thing:

    random play        the heuristic discard with the cards thrown down at random
    greedy             one-ply, but with the opponent's reply ignored entirely
    best reply         one-ply, assuming the opponent finds their best card
    two ply            searching one move further

Plus a random *discard* control, so the two halves of a cribbage policy can be
weighed against each other.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import DEFAULT_WORKERS, AgentSpec, Table, compare  # noqa: E402

from cribbage.agents import (  # noqa: E402
    HeuristicAgent, PeggingAwareAgent, PimcPeggingAgent,
    RandomDiscardAgent, RandomPlayAgent,
)

BASELINE = AgentSpec("heuristic", HeuristicAgent)


def pimc(args) -> int:
    """Exact search against sampled holdings, versus searching shallowly.

    The play is a small game once the opponent's hand is known -- eight plies,
    four choices a node -- so the interesting question is whether solving it
    outright against guessed holdings beats reasoning two plies deep against a
    marginal.
    """
    started = time.time()
    table = Table(width=26)
    pairs = args.pimc_pairs
    print(f"{pairs} pairs = {pairs * 2} games per row. This is the slow one: "
          f"exact search runs about 2 games a second a core.")

    deep = {"peg_weight": 0.5, "samples": 24}
    table.header("Exact play search")
    table.row(compare(AgentSpec("pimc, 24 samples", PimcPeggingAgent, deep),
                      BASELINE, pairs, args.seed + 11, args.workers))
    table.row(compare(
        AgentSpec("pimc, 24 samples", PimcPeggingAgent, deep),
        AgentSpec("twoply", PeggingAwareAgent,
                  {"peg_weight": 0.5, "play_depth": 2}),
        pairs, args.seed + 22, args.workers,
    ))

    table.header("How many sampled holdings are enough?")
    for index, samples in enumerate([4, 8, 48]):
        table.row(compare(
            AgentSpec(f"pimc, {samples} samples", PimcPeggingAgent,
                      {"peg_weight": 0.5, "samples": samples}),
            AgentSpec("pimc, 24 samples", PimcPeggingAgent, deep),
            pairs // 2, args.seed + 33 + index * 104729, args.workers,
        ))

    table.footer(time.time() - started)
    return 0


def confirm(args) -> int:
    """Fresh seeds for the depth result, and the two improvements combined."""
    started = time.time()
    table = Table(width=26)
    print(f"Fresh seeds, {args.depth_pairs} pairs = {args.depth_pairs * 2} games "
          f"per row.")

    table.header("Depth, and depth plus the pegging-aware discard")
    rows = [
        ("2-ply play alone", BASELINE, {"peg_weight": 0.0, "play_depth": 2}),
        ("pegging discard alone", BASELINE, {"peg_weight": 0.5, "play_depth": 1}),
        ("both", BASELINE, {"peg_weight": 0.5, "play_depth": 2}),
    ]
    for index, (label, against, kwargs) in enumerate(rows):
        table.row(compare(
            AgentSpec(label, PeggingAwareAgent, kwargs), against,
            args.depth_pairs, args.seed + 60013 + index * 104729, args.workers,
        ))

    table.header("Head to head: is the combination better than either alone?")
    table.row(compare(
        AgentSpec("both", PeggingAwareAgent, {"peg_weight": 0.5, "play_depth": 2}),
        AgentSpec("2-ply only", PeggingAwareAgent,
                  {"peg_weight": 0.0, "play_depth": 2}),
        args.depth_pairs, args.seed + 90007, args.workers,
    ))

    table.footer(time.time() - started)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--pairs", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=8080)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--depth-pairs", type=int, default=2500)
    parser.add_argument("--pimc", action="store_true",
                        help="solve the play exactly against sampled opponent "
                             "holdings, and compare against the shallower agents")
    parser.add_argument("--pimc-pairs", type=int, default=2000)
    parser.add_argument("--confirm", action="store_true",
                        help="replicate the depth result on fresh seeds and "
                             "combine it with the pegging-aware discard")
    args = parser.parse_args()

    if args.confirm:
        return confirm(args)
    if args.pimc:
        return pimc(args)

    started = time.time()
    table = Table(width=24)
    print(f"Mirrored matches against `{BASELINE.label}`, "
          f"{args.pairs} pairs = {args.pairs * 2} games each.")

    table.header("How much does each half of the policy matter?")
    rungs = [
        AgentSpec("random play", RandomPlayAgent, {"seed": 1}),
        AgentSpec("random discard", RandomDiscardAgent, {"seed": 1}),
        AgentSpec("greedy (no reply term)", HeuristicAgent, {"risk_weight": 0.0}),
        AgentSpec("best-reply model", PeggingAwareAgent,
                  {"peg_weight": 0.0, "best_reply": True}),
    ]
    for index, spec in enumerate(rungs):
        table.row(compare(spec, BASELINE, args.pairs,
                          args.seed + index * 104729, args.workers))

    table.header(f"Does looking deeper help? ({args.depth_pairs} pairs)")
    for index, depth in enumerate([2]):
        table.row(compare(
            AgentSpec(f"{depth}-ply play", PeggingAwareAgent,
                      {"peg_weight": 0.0, "play_depth": depth}),
            BASELINE, args.depth_pairs,
            args.seed + 5000 + index * 104729, args.workers,
        ))

    table.footer(time.time() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
