#!/usr/bin/env python3
"""Regenerate cribbage/data/crib_ev.json.

Exact enumeration -- every opponent lay-away and every starter for each of the
169 canonical discards.  Takes a while; it only needs running if the scoring
rules change.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cribbage.crib_table import DATA_FILE, compute_crib_table  # noqa: E402


def main() -> None:
    started = time.time()

    def progress(done: int, total: int) -> None:
        elapsed = time.time() - started
        rate = done / elapsed if elapsed else 0
        remaining = (total - done) / rate if rate else 0
        print(
            f"\r  {done:3d}/{total} entries  {elapsed:5.1f}s elapsed"
            f"  ~{remaining:5.1f}s remaining",
            end="",
            flush=True,
        )

    print("Enumerating crib expectations for all 169 canonical lay-aways...")
    table = compute_crib_table(progress=progress)
    print()

    out = Path(__file__).resolve().parent.parent / "cribbage" / "data" / DATA_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: round(v, 6) for k, v in sorted(table.items())}, indent=0))

    best = max(table.items(), key=lambda kv: kv[1])
    worst = min(table.items(), key=lambda kv: kv[1])
    print(f"Wrote {len(table)} entries to {out} in {time.time() - started:.1f}s")
    print(f"  richest lay-away:  {best[0]} -> {best[1]:.3f}")
    print(f"  poorest lay-away:  {worst[0]} -> {worst[1]:.3f}")


if __name__ == "__main__":
    main()
