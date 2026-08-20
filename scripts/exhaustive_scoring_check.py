#!/usr/bin/env python3
"""Exhaustively verify hand scoring against a property nothing else can fake.

Every five-card combination, with each of its five cards taken in turn as the
starter: 2,598,960 x 5 = 12,994,800 scored hands.  Two things are asserted.

**No hand can total 19, 25, 26 or 27.**  This is a well-known fact about
cribbage that falls out of the interaction between fifteens, pairs and runs --
"nineteen" is the traditional joke name for a zero-point hand.  It is a strong
property because it constrains the *whole* scorer at once: an off-by-one
anywhere in fifteens, pairs, runs, flush or nobs would almost certainly produce
one of these totals somewhere in thirteen million hands.

**The maximum is 29.**  And it is reached only by the famous hand: three fives
plus the jack matching a five turned as the starter.

Slow on purpose.  Run it when the scoring rules change, not in the normal suite.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cribbage.cards import hand_str  # noqa: E402
from cribbage.scoring import score_hand  # noqa: E402

IMPOSSIBLE = (19, 25, 26, 27)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--crib", action="store_true", help="check crib scoring instead of hand scoring"
    )
    args = parser.parse_args()

    histogram: Counter[int] = Counter()
    best = (-1, None)
    started = time.time()
    combos = 0

    for five in combinations(range(52), 5):
        combos += 1
        for index in range(5):
            starter = five[index]
            hand = five[:index] + five[index + 1 :]
            total = score_hand(hand, starter, is_crib=args.crib)
            histogram[total] += 1
            if total > best[0]:
                best = (total, (hand, starter))

        if combos % 200_000 == 0:
            elapsed = time.time() - started
            done = combos / 2_598_960
            print(
                f"\r  {done:5.1%}  {elapsed:5.1f}s elapsed"
                f"  ~{elapsed / done - elapsed:5.1f}s remaining",
                end="", flush=True,
            )

    print("\r" + " " * 60 + "\r", end="")
    scored = sum(histogram.values())
    kind = "crib" if args.crib else "hand"
    print(f"Scored {scored:,} {kind}s in {time.time() - started:.1f}s\n")

    failures = [total for total in IMPOSSIBLE if histogram[total]]
    for total in IMPOSSIBLE:
        count = histogram[total]
        mark = "FAIL" if count else "ok"
        print(f"  {kind}s totalling {total:>2}: {count:>10,}   [{mark}]")

    top = max(histogram)
    print(f"\n  maximum total: {top}  ({histogram[top]:,} hands)")
    if best[1] is not None:
        hand, starter = best[1]
        print(f"  example: {hand_str(hand)} + {hand_str([starter])}")

    print("\n  most common totals:")
    for total, count in sorted(histogram.items(), key=lambda kv: -kv[1])[:6]:
        print(f"    {total:>2} points: {count:>10,}  ({count / scored:6.2%})")

    if failures:
        print(f"\nFAILED: reachable totals that should be impossible: {failures}")
        return 1
    if not args.crib and top != 29:
        print(f"\nFAILED: maximum hand should be 29, got {top}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
