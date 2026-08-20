#!/usr/bin/env python3
"""Measure how much position awareness and defensive discarding are worth.

Every comparison is a *mirrored* match: each deal sequence is played twice with
the seats swapped, so both agents hold both sides of every hand.  Cribbage deal
luck is large -- roughly 95% of pairs split -- so without this the effects being
measured here would be buried in noise.

Confidence intervals are computed on per-pair scores rather than per-game
results, since the two games of a pair are strongly correlated.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cribbage.agents import HeuristicAgent, PositionalAgent  # noqa: E402
from cribbage.arena import PairedResult, run_paired_match  # noqa: E402


@dataclass
class AgentSpec:
    label: str
    cls: type
    kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self):
        return self.cls(**self.kwargs)


BASELINE = AgentSpec("heuristic", HeuristicAgent)

# Everything off reproduces HeuristicAgent exactly, so each variant below
# differs from the baseline in exactly one respect.
OFF = dict(endgame=False, finish=False)

BEST = dict(
    endgame=True, finish=True,
    stance_variance=0.3, stance_crib=0.4, stance_peg=0.5, crib_defense=0.25,
)


def variants() -> list[AgentSpec]:
    return [
        AgentSpec("endgame", PositionalAgent, {**OFF, "endgame": True}),
        AgentSpec("stance-variance", PositionalAgent, {**OFF, "stance_variance": 0.3}),
        AgentSpec("stance-crib", PositionalAgent, {**OFF, "stance_crib": 0.4}),
        AgentSpec("stance-peg", PositionalAgent, {**OFF, "stance_peg": 0.5}),
        AgentSpec("crib-defense", PositionalAgent, {**OFF, "crib_defense": 0.25}),
        AgentSpec("five-penalty", PositionalAgent, {**OFF, "five_penalty": 1.0}),
        AgentSpec("finish", PositionalAgent, {**OFF, "finish": True}),
        AgentSpec("all-combined", PositionalAgent, BEST),
    ]


# The endgame objective trades expected points for the chance of going out this
# deal.  `point_value` sets that exchange rate, so sweeping it separates "the
# idea is wrong" from "the exchange rate is wrong".
ENDGAME_SWEEP = [0.006, 0.012, 0.025, 0.05, 0.10]

SWEEPS = [
    ("How much to swing variance with the lead", "stance_variance",
     [0.15, 0.3, 0.6, 1.2]),
    ("How hard to defend the opponent's crib", "crib_defense",
     [0.25, 0.5, 1.0, 2.0, 4.0]),
]


def parse_spec(text: str) -> dict[str, Any]:
    """Turn ``"stance_variance=0.2,endgame=1"`` into agent keyword arguments.

    Values that name a boolean flag are coerced to bool, everything else to
    float, so ``endgame=1`` and ``stance_variance=0.2`` both do the obvious
    thing.
    """
    flags = {"endgame", "endgame_smoothing", "finish"}
    kwargs: dict[str, Any] = {}
    for part in text.split(","):
        if not part.strip():
            continue
        knob, _, value = part.partition("=")
        knob = knob.strip()
        if not _:
            raise ValueError(f"expected knob=value, got {part!r}")
        kwargs[knob] = bool(float(value)) if knob in flags else float(value)
    return kwargs


def _run_chunk(args) -> PairedResult:
    spec_a, spec_b, pairs, seed = args
    return run_paired_match(
        spec_a.build, spec_b.build, pairs=pairs, seed=seed,
        name_a=spec_a.label, name_b=spec_b.label,
    )


def compare(spec_a: AgentSpec, spec_b: AgentSpec, pairs: int, seed: int, workers: int):
    """Run one mirrored comparison, split across processes."""
    per_worker = [pairs // workers] * workers
    for i in range(pairs % workers):
        per_worker[i] += 1
    jobs = [
        (spec_a, spec_b, count, seed + index * 7919)
        for index, count in enumerate(per_worker)
        if count
    ]
    if workers == 1:
        results = [_run_chunk(job) for job in jobs]
    else:
        with Pool(workers) as pool:
            results = pool.map(_run_chunk, jobs)
    merged = results[0]
    for extra in results[1:]:
        merged.merge(extra)
    return merged


def print_row(result: PairedResult) -> None:
    global COMPARISONS
    COMPARISONS += 1
    low, high = result.interval
    delta = result.win_rate - 0.5
    if result.significant:
        verdict = "yes" if delta > 0 else "WORSE"
    else:
        verdict = "-"
    print(
        f"  {result.name_a:<20}{result.win_rate:7.2%}  "
        f"[{low:6.2%}, {high:6.2%}]  {delta:+6.2%}   {verdict:<6}"
        f"  {result.swept_a:>5} {result.swept_b:>5} {result.split:>6}"
    )


COMPARISONS = 0


def header(title: str) -> None:
    print(f"\n{title}")
    print(
        f"  {'variant':<20}{'win rate':>7}  {'95% interval':^16}  {'delta':>6}"
        f"   {'sig?':<6}  {'swept':>5} {'lost':>5} {'split':>6}"
    )
    print("  " + "-" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--pairs", type=int, default=2500,
                        help="mirrored deal sequences per comparison (2 games each)")
    parser.add_argument("--sweep-pairs", type=int, default=1500)
    parser.add_argument("--crib-defense", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--endgame-only", action="store_true",
                        help="only run the endgame exchange-rate sweep")
    parser.add_argument("--variant", action="append", default=[], metavar="SPEC",
                        help="test one ad-hoc config, e.g. "
                             "--variant stance_variance=0.2,endgame=1 . Repeatable. "
                             "Use for confirming a promising result on fresh seeds.")
    args = parser.parse_args()

    started = time.time()
    print(
        f"Mirrored matches against `{BASELINE.label}`, "
        f"{args.pairs} pairs = {args.pairs * 2} games each, "
        f"{args.workers} workers."
    )

    if args.variant:
        header(f"Ad-hoc variants ({args.pairs} pairs)")
        for index, text in enumerate(args.variant):
            spec = AgentSpec(text[:20], PositionalAgent, {**OFF, **parse_spec(text)})
            result = compare(
                spec, BASELINE, args.pairs, args.seed + index * 104729, args.workers
            )
            print_row(result)
        footer(started)
        return 0

    if args.endgame_only:
        header(
            "Endgame exchange rate: win-probability value of one point "
            f"({args.pairs} pairs)"
        )
        for index, value in enumerate(ENDGAME_SWEEP):
            spec = AgentSpec(
                f"point_value={value}", PositionalAgent,
                {**OFF, "endgame": True, "point_value": value},
            )
            result = compare(
                spec, BASELINE, args.pairs, args.seed + index * 104729, args.workers
            )
            print_row(result)
        footer(started)
        return 0

    header("Feature ablation (each variant differs from baseline in one respect)")
    for index, spec in enumerate(variants()):
        result = compare(spec, BASELINE, args.pairs, args.seed + index * 104729, args.workers)
        print_row(result)

    if not args.skip_sweep:
        offset = 500
        for title, knob, values in SWEEPS:
            header(f"{title} ({args.sweep_pairs} pairs, other features off)")
            for index, weight in enumerate(values):
                spec = AgentSpec(
                    f"{knob}={weight}", PositionalAgent, {**OFF, knob: weight},
                )
                result = compare(
                    spec, BASELINE, args.sweep_pairs,
                    args.seed + offset + index * 104729, args.workers,
                )
                print_row(result)
            offset += 10000

    footer(started)
    return 0


def footer(started: float) -> None:
    """Say plainly how much of any 'significant' result is expected to be noise."""
    expected_false = COMPARISONS * 0.05
    print(
        f"\n{COMPARISONS} comparisons at 95% confidence, so about "
        f"{expected_false:.1f} of them are expected to clear significance by "
        f"chance alone. Treat a single marginal winner as a hypothesis and "
        f"re-run it on fresh seeds before believing it."
    )
    print(f"Total {time.time() - started:.0f}s")


if __name__ == "__main__":
    raise SystemExit(main())
