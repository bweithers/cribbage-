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
* **Variance shares** -- a three-way ANOVA on the final margin.  Reported as eta
  squared (raw share of sum of squares, which flatters factors with many levels),
  omega squared (adjusted, using the three-way interaction as the error term),
  an F test against that error term, and -- most interpretably -- the estimated
  standard deviation of each factor's effect *in points of final margin*.

The F column matters for reading the small numbers honestly.  A factor can be
statistically detectable and still be practically negligible, and the cut turns
out to be exactly that: the question "does it beat noise" and the question "does
it matter" have different answers.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

try:
    from scipy import stats as _stats
except ImportError:  # pragma: no cover - p-values are a nicety, not a requirement
    _stats = None

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


def decompose(y: np.ndarray) -> dict[str, dict[str, float]]:
    """Three-way ANOVA on a (deal, cut, dealer) grid with one game per cell.

    With no replication the three-way interaction stands in as the error term;
    in a deterministic game it is not noise but genuine irreducible interaction
    between the specific cards, the specific cuts and who dealt.

    For each source returns eta squared, omega squared, degrees of freedom, the
    F ratio against that error term, its p-value, and ``sigma`` -- the estimated
    standard deviation of the factor's effect in the units of ``y``.  ``sigma``
    is the one to quote: a share of variance says nothing about whether the
    effect is worth caring about, and points of margin do.
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
    cells = y.size
    # Observations averaged over when estimating each main effect's level means.
    per_level = {"deal": b * c, "cut": a * c, "dealt first": a * b}

    out: dict[str, dict[str, float]] = {}
    for name, value in ss.items():
        mean_square = value / df[name]
        eta = value / total
        omega = max(0.0, value - df[name] * ms_error) / (total + ms_error)
        ratio = mean_square / ms_error if name != "three-way" else float("nan")
        if name == "three-way" or _stats is None:
            p_value = float("nan")
        else:
            p_value = float(_stats.f.sf(ratio, df[name], df["three-way"]))
        # Random-effects variance component, floored at zero.
        divisor = per_level.get(name, cells / (df[name] + 1))
        component = max(0.0, (mean_square - ms_error) / divisor)
        out[name] = {
            "eta": eta, "omega": omega, "df": df[name],
            "F": ratio, "p": p_value, "sigma": component ** 0.5,
        }
    return out


def flip_rate(win: np.ndarray, axis: int) -> float:
    """Chance the winner changes when only ``axis`` is re-randomized.

    For a fixed setting of the other factors the outcome is Bernoulli(p) across
    that axis, so two independent draws disagree with probability 2p(1-p).
    Computed from the marginal rather than by sampling pairs, so there is no
    Monte Carlo error beyond the estimate of p itself.

    The plug-in estimator is biased low: E[2*phat*(1-phat)] = 2p(1-p)(1 - 1/n),
    because phat is estimated from the same n draws.  The n/(n-1) factor undoes
    it -- worth about a percent of the reported figure at n=128.
    """
    n = win.shape[axis]
    p = win.mean(axis=axis)
    return float((2 * p * (1 - p)).mean() * n / (n - 1))


def starter_report(games: int, seed: int) -> None:
    """Does a *good* cut favour the dealer?  The ANOVA above cannot answer this.

    Its `cut` factor is a seed for an index sequence, which is seat-neutral by
    construction: whether index 7 is a good card depends on what was dealt, so
    no cut seed can persistently favour a seat and the main effect is near zero
    almost tautologically.  The question worth asking is about the *card* that
    came up, not the seed that chose it -- and the dealer scores the starter
    against two holdings (hand and crib) where the pone scores it against one.
    """
    import random
    from collections import defaultdict

    from cribbage.cards import RANK_CHARS, rank_of

    by_rank: dict[int, list[list[int]]] = defaultdict(lambda: [[], []])
    rng = random.Random(seed)
    for index in range(games):
        state = play_game(
            [make_agent(AGENT), make_agent(AGENT)],
            seed=rng.randrange(1 << 30), dealer=index % 2,
        )
        first_dealer = state.events[0].player
        last = state.round_index
        starters: dict[int, int] = {}
        points: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for event in state.events:
            if event.round == last:
                continue  # truncated by the win
            if event.kind == "cut":
                starters[event.round] = int(event.detail and _card_from(event.detail))
            elif event.points:
                dealt = event.player == (first_dealer + event.round - 1) % 2
                points[event.round][0 if dealt else 1] += event.points
        for round_index, starter in starters.items():
            if round_index in points:
                got = points[round_index]
                by_rank[rank_of(starter)][0].append(got[0])
                by_rank[rank_of(starter)][1].append(got[1])

    print("\nPoints per deal by the rank of the starter turned")
    print(f"  {'starter':<9}{'dealer':>8}{'pone':>8}{'gap':>8}{'deals':>9}")
    print("  " + "-" * 42)
    gaps = []
    for rank in sorted(by_rank):
        dealer_pts, pone_pts = by_rank[rank]
        gap = np.mean(dealer_pts) - np.mean(pone_pts)
        gaps.append((RANK_CHARS[rank], gap, len(dealer_pts)))
        print(
            f"  {RANK_CHARS[rank]:<9}{np.mean(dealer_pts):8.2f}"
            f"{np.mean(pone_pts):8.2f}{gap:8.2f}{len(dealer_pts):9d}"
        )
    best = max(gaps, key=lambda row: row[1])
    worst = min(gaps, key=lambda row: row[1])
    print(
        f"\n  The dealer's edge is widest on a {best[0]} ({best[1]:.2f} points) "
        f"and narrowest on a {worst[0]} ({worst[1]:.2f}).\n"
        f"  Spread of {best[1] - worst[1]:.2f} points across starter ranks -- so a "
        f"lucky cut does favour\n  the dealer, which the seed-level ANOVA is "
        f"structurally unable to see."
    )


def _card_from(text: str) -> int:
    from cribbage.cards import parse_card

    return parse_card(text.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-k", "--levels", type=int, default=96,
                        help="deal seeds and cut seeds; plays 2*K*K games")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--save", type=Path, default=None,
                        help="write the grid to a .npz so it can be re-analysed "
                             "without replaying the games")
    parser.add_argument("--load", type=Path, default=None,
                        help="re-analyse a grid saved by --save")
    parser.add_argument("--starters", type=int, default=0, metavar="GAMES",
                        help="also report points per deal by starter rank, which "
                             "the seed-level ANOVA structurally cannot measure")
    args = parser.parse_args()

    if args.load:
        cached = np.load(args.load)
        margin, win = cached["margin"], cached["win"]
        print(f"Loaded a {margin.shape[0]}x{margin.shape[1]}x2 grid "
              f"({margin.size} games) from {args.load}.\n")
        code = report(margin, win)
        if args.starters:
            starter_report(args.starters, args.seed)
        return code

    games = 2 * args.levels ** 2
    print(
        f"{AGENT} vs {AGENT} (identical and deterministic, so skill contributes "
        f"nothing).\n{args.levels} deals x {args.levels} cuts x 2 dealers "
        f"= {games} games."
    )
    started = time.time()
    margin, win = build_grid(args.levels, args.seed, args.workers)
    print(f"Played in {time.time() - started:.0f}s.\n")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save, margin=margin, win=win)
        print(f"Saved the grid to {args.save}.\n")
    code = report(margin, win)
    if args.starters:
        starter_report(args.starters, args.seed)
    return code


def report(margin: np.ndarray, win: np.ndarray) -> int:
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
    for label, data, unit in (
        ("final margin", margin, "points"), ("win/loss", win, "win prob"),
    ):
        print(f"\nVariance of {label}, by source"
              f"   (total sd {data.std():.3f} {unit})")
        print(
            f"  {'source':<22}{'eta^2':>8}{'omega^2':>9}{'df':>7}"
            f"{'F':>8}{'p':>10}{'effect sd':>11}"
        )
        print("  " + "-" * 75)
        for name, row in decompose(data).items():
            f_text = "  (error)" if name == "three-way" else f"{row['F']:8.2f}"
            p_text = "" if name == "three-way" else _format_p(row["p"])
            print(
                f"  {name:<22}{row['eta']:8.1%}{row['omega']:9.1%}"
                f"{row['df']:7d}{f_text}{p_text:>10}{row['sigma']:9.3f} {unit}"
            )

    print(
        "\neta^2 flatters `deal` and `cut`, which have many degrees of freedom, "
        "against\n`dealt first`, which has one.  The effect-sd column is the one "
        "to quote: it is\nin points, so it says whether a factor matters rather "
        "than only whether it is\ndetectable."
    )
    return 0


def _format_p(value: float) -> str:
    if value != value:  # NaN
        return "n/a"
    if value < 1e-12:
        return "<1e-12"
    return f"{value:.2g}"


if __name__ == "__main__":
    raise SystemExit(main())
