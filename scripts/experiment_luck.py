#!/usr/bin/env python3
"""Decompose cribbage luck into its sources.

Two agents that play identically remove skill from the picture entirely, and the
agents here are deterministic, so a finished game is a pure function of three
independent inputs:

    outcome = f(deal, cut, who dealt first)

That makes the variance decomposition *exact* -- there is no residual noise term
to explain away.  The engine draws the deal and the cut from separate RNG
streams (``seed`` and ``cut_seed``), and the deck always holds exactly 40 cards
when the starter is turned, so a given cut stream draws the same index every
round no matter what was dealt.  Holding one factor and re-randomizing another
is therefore proper common random numbers, not an approximation.

The design is a full factorial: K deal seeds x K cut seeds x 2 dealers.  Equal
level counts for deal and cut keep their degrees of freedom comparable, so their
variance shares can be read against each other.

Two views of the same grid:

* **Flip rates** -- "if we replayed this game and re-randomized only X, how often
  would the winner change?"  Directly interpretable, and computed exactly as
  ``2p(1-p)`` rather than by sampling pairs.
* **Variance shares** -- a three-way ANOVA on the final margin.  Reported both as
  eta squared (raw share of sum of squares, which flatters factors with many
  levels) and omega squared (adjusted, using the three-way interaction as the
  error term).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cribbage.agents import make_agent  # noqa: E402
from cribbage.arena import play_game  # noqa: E402

AGENT = "heuristic"


def _worker(args):
    rows, deal_seeds, cut_seeds = args
    out = []
    for i in rows:
        for j, cut in enumerate(cut_seeds):
            for dealer in (0, 1):
                state = play_game(
                    [make_agent(AGENT), make_agent(AGENT)],
                    seed=deal_seeds[i], cut_seed=cut, dealer=dealer,
                )
                result = state.result()
                out.append(
                    (i, j, dealer,
                     result.scores[0] - result.scores[1],
                     1 if result.winner == 0 else 0)
                )
    return out


def build_grid(levels: int, seed: int, workers: int):
    """Play the full factorial and return margin and win arrays, both (K, K, 2)."""
    rng = np.random.default_rng(seed)
    deal_seeds = rng.integers(0, 1 << 30, size=levels).tolist()
    cut_seeds = rng.integers(0, 1 << 30, size=levels).tolist()

    chunks = [list(range(i, levels, workers)) for i in range(workers)]
    jobs = [(chunk, deal_seeds, cut_seeds) for chunk in chunks if chunk]

    if workers == 1:
        results = [_worker(job) for job in jobs]
    else:
        with Pool(workers) as pool:
            results = pool.map(_worker, jobs)

    margin = np.zeros((levels, levels, 2))
    win = np.zeros((levels, levels, 2))
    for batch in results:
        for i, j, dealer, m, w in batch:
            margin[i, j, dealer] = m
            win[i, j, dealer] = w
    return margin, win


def decompose(y: np.ndarray) -> dict[str, tuple[float, float, int]]:
    """Three-way ANOVA on a (deal, cut, dealer) grid with one game per cell.

    Returns name -> (eta squared, omega squared, degrees of freedom).  With no
    replication the three-way interaction stands in as the error term; in a
    deterministic game it is not noise but genuine irreducible interaction
    between the specific cards, the specific cuts and who dealt.
    """
    a, b, c = y.shape
    grand = y.mean()
    total = ((y - grand) ** 2).sum()

    mean_a = y.mean(axis=(1, 2))
    mean_b = y.mean(axis=(0, 2))
    mean_c = y.mean(axis=(0, 1))
    ss = {
        "deal": b * c * ((mean_a - grand) ** 2).sum(),
        "cut": a * c * ((mean_b - grand) ** 2).sum(),
        "dealt first": a * b * ((mean_c - grand) ** 2).sum(),
    }
    ss["deal x cut"] = c * (
        (y.mean(axis=2) - mean_a[:, None] - mean_b[None, :] + grand) ** 2
    ).sum()
    ss["deal x dealt first"] = b * (
        (y.mean(axis=1) - mean_a[:, None] - mean_c[None, :] + grand) ** 2
    ).sum()
    ss["cut x dealt first"] = a * (
        (y.mean(axis=0) - mean_b[:, None] - mean_c[None, :] + grand) ** 2
    ).sum()
    ss["three-way"] = total - sum(ss.values())

    df = {
        "deal": a - 1, "cut": b - 1, "dealt first": c - 1,
        "deal x cut": (a - 1) * (b - 1),
        "deal x dealt first": (a - 1) * (c - 1),
        "cut x dealt first": (b - 1) * (c - 1),
        "three-way": (a - 1) * (b - 1) * (c - 1),
    }
    ms_error = ss["three-way"] / df["three-way"]

    out = {}
    for name, value in ss.items():
        eta = value / total
        omega = max(0.0, value - df[name] * ms_error) / (total + ms_error)
        out[name] = (eta, omega, df[name])
    return out


def flip_rate(win: np.ndarray, axis: int) -> float:
    """Chance the winner changes when only ``axis`` is re-randomized.

    For a fixed setting of the other factors the outcome is Bernoulli(p) across
    that axis, so two independent draws disagree with probability 2p(1-p).  No
    sampling needed.
    """
    p = win.mean(axis=axis)
    return float((2 * p * (1 - p)).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-k", "--levels", type=int, default=96,
                        help="deal seeds and cut seeds; plays 2*K*K games")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    games = 2 * args.levels ** 2
    print(
        f"{AGENT} vs {AGENT} (identical and deterministic, so skill contributes "
        f"nothing).\n{args.levels} deals x {args.levels} cuts x 2 dealers "
        f"= {games} games."
    )
    started = time.time()
    margin, win = build_grid(args.levels, args.seed, args.workers)
    print(f"Played in {time.time() - started:.0f}s.\n")

    # ---- Who deals first ------------------------------------------------
    first_dealer_wins = np.concatenate([win[:, :, 0].ravel(), 1 - win[:, :, 1].ravel()])
    rate = first_dealer_wins.mean()
    # Cluster the interval on the deal, since cells sharing a deal are related.
    per_deal = np.concatenate([win[:, :, 0], 1 - win[:, :, 1]], axis=1).mean(axis=1)
    half = 1.96 * per_deal.std(ddof=1) / np.sqrt(len(per_deal))
    print("Who deals first")
    print(f"  first dealer wins        {rate:6.2%}  "
          f"(95% CI {rate - half:.2%}-{rate + half:.2%}, clustered by deal)")
    swapped = (win[:, :, 0] != win[:, :, 1]).mean()
    print(f"  swapping only the deal flips the winner  {swapped:6.2%}")
    print(f"  mean margin swing from the deal          "
          f"{(margin[:, :, 0] - margin[:, :, 1]).mean():+6.2f} points\n")

    # ---- Flip rates -----------------------------------------------------
    print("If we replayed the game and re-randomized only...")
    rows = [
        ("the cards dealt", flip_rate(win, 0)),
        ("the cut cards", flip_rate(win, 1)),
        ("who deals first", float(swapped)),
        ("everything", float(2 * win.mean() * (1 - win.mean()))),
    ]
    for label, value in rows:
        bar = "#" * int(round(value * 100))
        print(f"  {label:<20} winner changes {value:6.2%}  {bar}")

    # ---- Variance shares ------------------------------------------------
    for label, data in (("final margin", margin), ("win/loss", win)):
        print(f"\nVariance of {label}, by source")
        print(f"  {'source':<22}{'eta^2':>8}{'omega^2':>10}{'df':>8}")
        print("  " + "-" * 48)
        for name, (eta, omega, degrees) in decompose(data).items():
            print(f"  {name:<22}{eta:8.1%}{omega:10.1%}{degrees:8d}")

    print(
        "\nomega^2 is the honest column: eta^2 flatters `deal` and `cut` because "
        "they have\nmany levels, while `dealt first` has one degree of freedom."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
