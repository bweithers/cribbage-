"""Shared machinery for mirrored agent comparisons.

Every experiment in this directory asks the same question -- is variant A better
than baseline B -- and answers it the same way, so the answering lives here.

Mirrored matches: each deal sequence is played twice with the seats swapped, so
both agents hold both sides of every hand.  Cribbage deal luck is large enough
that without this the effects being measured are buried, and the intervals are
computed on per-pair scores rather than per-game results because the two games
of a pair are strongly correlated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Any

from cribbage.arena import PairedResult, run_paired_match

DEFAULT_WORKERS = os.cpu_count() or 1


@dataclass
class AgentSpec:
    """A named agent configuration, picklable so it can cross into a worker."""

    label: str
    cls: type
    kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self):
        return self.cls(**self.kwargs)


def _run_chunk(job) -> PairedResult:
    spec_a, spec_b, pairs, seed = job
    return run_paired_match(
        spec_a.build, spec_b.build, pairs=pairs, seed=seed,
        name_a=spec_a.label, name_b=spec_b.label,
    )


def compare(spec_a: AgentSpec, spec_b: AgentSpec, pairs: int, seed: int,
            workers: int = DEFAULT_WORKERS) -> PairedResult:
    """One mirrored comparison, split across processes."""
    per_worker = [pairs // workers] * workers
    for index in range(pairs % workers):
        per_worker[index] += 1
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


class Table:
    """Prints comparison rows, and counts them so the footer can be honest."""

    def __init__(self, width: int = 20):
        self.width = width
        self.rows = 0

    def header(self, title: str) -> None:
        print(f"\n{title}")
        print(f"  {'variant':<{self.width}}{'win rate':>8}  {'95% interval':^17}"
              f"  {'delta':>7}   {'sig?':<6}  {'swept':>5} {'lost':>5} {'split':>6}")
        print("  " + "-" * (self.width + 58))

    def row(self, result: PairedResult) -> None:
        self.rows += 1
        low, high = result.interval
        delta = result.win_rate - 0.5
        verdict = ("yes" if delta > 0 else "WORSE") if result.significant else "-"
        print(
            f"  {result.name_a:<{self.width}}{result.win_rate:8.2%}  "
            f"[{low:6.2%}, {high:6.2%}]  {delta:+7.2%}   {verdict:<6}"
            f"  {result.swept_a:>5} {result.swept_b:>5} {result.split:>6}"
        )

    def footer(self, elapsed: float) -> None:
        print(
            f"\n{self.rows} comparisons at 95% confidence, so roughly "
            f"{self.rows * 0.05:.1f} of them are expected\nto clear significance "
            f"by chance alone. Re-run a single marginal winner on fresh\nseeds "
            f"before believing it."
        )
        print(f"Total {elapsed:.0f}s")
