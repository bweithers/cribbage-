#!/usr/bin/env python3
"""Is a pegging-aware discard worth anything?

The position experiment came back null across the board, and the diagnosis was
that those features perturbed only one to three percent of discards.  This one
changes about fifteen percent of them, because it addresses a term the baseline
objective omits entirely rather than nudging one it already has.

Measured the same way as everything else here: mirrored deals, intervals on
per-pair scores, and a weight sweep so the answer is not hostage to one guess
about how much to trust the new term.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _harness import DEFAULT_WORKERS, AgentSpec, Table, compare  # noqa: E402

from cribbage.agents import HeuristicAgent, PeggingAwareAgent  # noqa: E402

BASELINE = AgentSpec("heuristic", HeuristicAgent)


def confirm(args) -> int:
    """Re-test on fresh seeds, and separate the discard term from the play model.

    The sweep below found its best weight by looking at the data, so quoting that
    weight's win rate from the same run would be reporting the noise it was
    selected on.  These are independent seeds.
    """
    started = time.time()
    table = Table(width=26)
    print(f"Fresh seeds, {args.pairs} pairs = {args.pairs * 2} games per row.")

    discard_only = {"best_reply": False}
    table.header("Replication, and the two halves separated")
    rows = [
        ("discard term w=0.5", {"peg_weight": 0.5, **discard_only}),
        ("discard term w=1.0", {"peg_weight": 1.0, **discard_only}),
        ("best-reply play only", {"peg_weight": 0.0, "best_reply": True}),
        ("both (w=0.5)", {"peg_weight": 0.5, "best_reply": True}),
    ]
    for index, (label, kwargs) in enumerate(rows):
        table.row(compare(
            AgentSpec(label, PeggingAwareAgent, kwargs), BASELINE,
            args.pairs, args.seed + 7717 + index * 104729, args.workers,
        ))

    table.footer(time.time() - started)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--pairs", type=int, default=3000)
    parser.add_argument("--sweep-pairs", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=31337)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--confirm", action="store_true",
                        help="replicate the winners on fresh seeds, and measure "
                             "the best-reply play model")
    args = parser.parse_args()

    if args.confirm:
        return confirm(args)

    started = time.time()
    table = Table()
    print(f"Mirrored matches against `{BASELINE.label}`, "
          f"{args.pairs} pairs = {args.pairs * 2} games each, "
          f"{args.workers} workers.")

    table.header("Pegging-aware discard, at the principled weight of 1.0")
    table.row(compare(
        AgentSpec("pegging (w=1.0)", PeggingAwareAgent),
        BASELINE, args.pairs, args.seed, args.workers,
    ))

    table.header(f"How much to trust the pegging term ({args.sweep_pairs} pairs)")
    for index, weight in enumerate([0.25, 0.5, 1.5, 2.0, 3.0]):
        table.row(compare(
            AgentSpec(f"w={weight}", PeggingAwareAgent, {"peg_weight": weight}),
            BASELINE, args.sweep_pairs, args.seed + 991 + index * 104729,
            args.workers,
        ))

    table.footer(time.time() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
