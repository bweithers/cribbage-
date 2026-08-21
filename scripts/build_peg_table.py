#!/usr/bin/env python3
"""Regenerate cribbage/data/peg_ev.json.

Estimates, for every four-card rank multiset, the pegging points a keep is worth
net of the opponent's -- once as dealer and once as pone.  Sampled rather than
exact, so it takes a while; see cribbage/peg_table.py for why the sampling is
arranged the way it is.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cribbage.peg_table import (  # noqa: E402
    DATA_FILE, all_keys, estimate_entry, reference_discard, table_key,
)

SAMPLES = 400


def _worker(job):
    keys, samples = job
    from cribbage.agents import HeuristicAgent

    agent = HeuristicAgent()

    def discarder(six, is_dealer):
        return reference_discard(agent, six, is_dealer)

    out = []
    for ranks in keys:
        as_pone = estimate_entry(ranks, False, samples, agent, discarder)
        as_dealer = estimate_entry(ranks, True, samples, agent, discarder)
        out.append((table_key(ranks), round(as_pone, 4), round(as_dealer, 4)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--samples", type=int, default=SAMPLES)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N multisets, for a quick smoke test")
    args = parser.parse_args()

    keys = all_keys()
    if args.limit:
        keys = keys[: args.limit]

    print(f"{len(keys)} rank multisets x 2 seats x {args.samples} scenarios "
          f"on {args.workers} workers.")
    started = time.time()

    chunks = [keys[i::args.workers] for i in range(args.workers)]
    jobs = [(chunk, args.samples) for chunk in chunks if chunk]
    if args.workers == 1:
        results = [_worker(job) for job in jobs]
    else:
        with Pool(args.workers) as pool:
            results = pool.map(_worker, jobs)

    table = {}
    for batch in results:
        for key, pone, dealer in batch:
            table[key] = [pone, dealer]

    out = Path(__file__).resolve().parent.parent / "cribbage" / "data" / DATA_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(sorted(table.items())), indent=0))

    elapsed = time.time() - started
    print(f"Wrote {len(table)} entries to {out} in {elapsed / 60:.1f} min.\n")

    from cribbage.cards import RANK_CHARS

    def show(label, index, reverse):
        rows = sorted(table.items(), key=lambda kv: kv[1][index], reverse=reverse)[:6]
        print(f"  {label}")
        for key, value in rows:
            hand = " ".join(RANK_CHARS[int(r)] for r in key.split(","))
            print(f"    {hand:<12} pone {value[0]:+6.2f}   dealer {value[1]:+6.2f}")

    show("best pegging keeps (as dealer):", 1, True)
    show("worst pegging keeps (as dealer):", 1, False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
