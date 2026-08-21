"""Command line entry points: ``demo``, ``match`` and ``bench``."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional, Sequence

from .agents import AGENTS, make_agent
from .arena import play_game, run_match
from .cards import card_str
from .engine import DEFAULT_TARGET, CribbageState, new_game
from .interactive import play_interactive
from .luck import cut_luck

__all__ = ["main", "format_transcript"]


# ----------------------------------------------------------------------
# Transcript
# ----------------------------------------------------------------------

_INDENT = "    "


def format_transcript(state: CribbageState, names: Sequence[str]) -> str:
    """Render a finished game as a readable log.

    This is a god's-eye view -- it shows both hands -- because its job is to let
    a human check the engine against the rules, not to be shown to an agent.
    """
    lines: list[str] = []
    scores = [0, 0]
    current_round = None

    def label(player: int) -> str:
        return f"P{player} {names[player]}"

    for event in state.events:
        if event.round != current_round:
            current_round = event.round
            dealer_at = _dealer_for_round(state, event.round)
            lines.append("")
            lines.append(
                f"── Round {event.round} " + "─" * 30 + f"  dealer: {label(dealer_at)}"
            )

        kind = event.kind
        if kind == "deal":
            for part in event.detail.split(" | "):
                seat = int(part[1])
                lines.append(f"{_INDENT}{label(seat):<16} {part[3:]}")
        elif kind == "discard":
            lines.append(f"{_INDENT}{label(event.player):<16} lays away  {event.detail}")
        elif kind == "cut":
            lines.append(f"{_INDENT}{'cut':<16} {event.detail}")
        elif kind == "reset":
            lines.append(f"{_INDENT}{'':<16} -- count resets --")
        else:
            if event.points:
                scores[event.player] += event.points
            marker = f"+{event.points}" if event.points else ""
            running = f"({scores[0]}-{scores[1]})" if event.points else ""
            detail = event.detail
            if kind == "play":
                text = f"plays {detail}"
            elif kind == "go":
                text = detail
            elif kind in ("hand", "crib"):
                text = f"{kind}: {detail}"
            elif kind == "heels":
                text = detail
            else:
                text = detail
            lines.append(
                f"{_INDENT}{label(event.player):<16} {text:<52} {marker:>3} {running}"
            )

    lines.append("")
    result = state.result()
    verdict = f"{label(result.winner)} wins {result.scores[result.winner]}-{result.scores[result.loser]}"
    if result.double_skunk:
        verdict += "  (double skunk!)"
    elif result.skunk:
        verdict += "  (skunk)"
    lines.append(verdict)
    return "\n".join(line.rstrip() for line in lines)


def _dealer_for_round(state: CribbageState, round_index: int) -> int:
    """Recover who dealt a given round.

    The deal alternates every round, and ``state.dealer`` has moved on by the
    time the game ends, so walk it back from the first round instead.
    """
    first_dealer = state.events[0].player if state.events else 0
    return (first_dealer + round_index - 1) % 2


# ----------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------


def cmd_demo(args: argparse.Namespace) -> int:
    agents = [make_agent(args.p0, seed=args.seed), make_agent(args.p1, seed=args.seed)]
    state = play_game(agents, seed=args.seed, dealer=0, target=args.target)
    names = [agents[0].name, agents[1].name]
    print(format_transcript(state, names))
    return 0


def cmd_luck(args: argparse.Namespace) -> int:
    agents = [make_agent(args.p0, seed=args.seed), make_agent(args.p1, seed=args.seed)]
    state = play_game(agents, seed=args.seed, dealer=0, target=args.target)
    result = state.result()
    names = [f"P{i} {agents[i].name}" for i in (0, 1)]

    print(f"Cut luck — {names[0]} vs {names[1]}, seed {args.seed}")
    print(f"  final {result.scores[0]}-{result.scores[1]}, "
          f"{names[result.winner]} wins\n")

    luck = cut_luck(state, args.player)
    who = names[args.player]
    print(f"  From {who}'s side. 'net' is their show points minus their")
    print("  opponent's, measured against what an average cut would have given.\n")
    print(f"  {'round':>6}{'cut':>6}{'show':>7}{'exp':>7}{'gross':>8}"
          f"{'net':>8}{'sd':>7}{'z':>7}")
    print("  " + "-" * 56)
    for row in luck.rounds:
        print(
            f"  {row.round_index:>6}{card_str(row.starter):>6}"
            f"{row.actual_gross:>7.0f}{row.expected_gross:>7.1f}"
            f"{row.gross:>+8.1f}{row.net:>+8.1f}{row.sd_net:>7.1f}"
            f"{row.net_z:>+7.2f}"
        )
    print("  " + "-" * 56)
    print(f"\n  {who}: {luck.describe()}")
    print(f"  gross: {luck.gross:+.1f} points (sd {luck.gross_sd:.1f}, "
          f"z = {luck.gross_z:+.2f})")
    if luck.skipped:
        print(f"\n  {luck.skipped} round(s) excluded: the game ended before the show.")
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    opponent = make_agent(args.opponent, seed=args.seed)
    play_interactive(
        opponent, seed=args.seed, human=args.seat, dealer=args.dealer,
        target=args.target, percentile=args.luck,
    )
    return 0


def cmd_match(args: argparse.Namespace) -> int:
    agents = [make_agent(args.p0, seed=args.seed), make_agent(args.p1, seed=args.seed + 1)]

    step = max(1, args.games // 100)

    def progress(done: int, total: int) -> None:
        if done % step == 0 or done == total:
            print(f"\r  {done}/{total} games", end="", file=sys.stderr, flush=True)

    started = time.perf_counter()
    stats = run_match(agents, games=args.games, seed=args.seed, target=args.target,
                      progress=None if args.quiet else progress)
    elapsed = time.perf_counter() - started
    if not args.quiet:
        print("\r" + " " * 30 + "\r", end="", file=sys.stderr)
    print(stats.report())
    print()
    print(f"  {args.games / elapsed:.1f} games/sec  ({elapsed:.1f}s total)")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .cards import parse_card, parse_hand
    from .scoring import score_hand, score_play

    print("scoring")
    hand, starter = parse_hand("4S 5H 5D 6C"), parse_card("6S")
    iterations = 300_000
    started = time.perf_counter()
    for _ in range(iterations):
        score_hand(hand, starter)
    per = (time.perf_counter() - started) / iterations
    print(f"  score_hand   {1 / per / 1e6:6.2f} M/sec   ({per * 1e6:.2f} us)")

    seq = parse_hand("3S 4H 2D 5C")
    started = time.perf_counter()
    for _ in range(iterations):
        score_play(seq)
    per = (time.perf_counter() - started) / iterations
    print(f"  score_play   {1 / per / 1e6:6.2f} M/sec   ({per * 1e6:.2f} us)")

    print("\ngames")
    for name in args.agents:
        agents = [make_agent(name, seed=1), make_agent(name, seed=2)]
        # Warm any lazily built tables before timing.
        play_game(agents, seed=0)
        started = time.perf_counter()
        stats = run_match(agents, games=args.games, seed=7)
        elapsed = time.perf_counter() - started
        print(
            f"  {name:<12} {args.games / elapsed:8.1f} games/sec"
            f"   ({elapsed / args.games * 1e3:6.2f} ms/game,"
            f" {stats.total_rounds / args.games:.1f} rounds/game)"
        )

    print("\nstate machine")
    started = time.perf_counter()
    total = 0
    for seed in range(args.games):
        state = new_game(seed=seed)
        while not state.is_terminal():
            state.apply_action(state.legal_actions()[0])
            total += 1
    elapsed = time.perf_counter() - started
    print(
        f"  apply_action {total / elapsed / 1e3:8.1f} K/sec"
        f"   ({total} decisions over {args.games} games)"
    )
    return 0


# ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cribbage", description="A two-player cribbage simulator."
    )
    known = ", ".join(sorted(AGENTS))
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="play one game and print a full transcript")
    demo.add_argument("--p0", default="heuristic", help=f"seat 0 agent ({known})")
    demo.add_argument("--p1", default="heuristic", help=f"seat 1 agent ({known})")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--target", type=int, default=DEFAULT_TARGET)
    demo.set_defaults(func=cmd_demo)

    match = sub.add_parser("match", help="play many games and report statistics")
    match.add_argument("--p0", default="heuristic", help=f"seat 0 agent ({known})")
    match.add_argument("--p1", default="random", help=f"seat 1 agent ({known})")
    match.add_argument("-n", "--games", type=int, default=1000)
    match.add_argument("--seed", type=int, default=0)
    match.add_argument("--target", type=int, default=DEFAULT_TARGET)
    match.add_argument("-q", "--quiet", action="store_true")
    match.set_defaults(func=cmd_match)

    play = sub.add_parser("play", help="play a game yourself in the terminal")
    play.add_argument("--opponent", default="heuristic", help=f"who to play ({known})")
    play.add_argument("--luck", type=float, default=None, metavar="PCT",
                      help="rig every cut to this percentile of what it could have "
                           "been for you (0 worst, 50 fair, 100 best)")
    play.add_argument("--seed", type=int, default=None)
    play.add_argument("--seat", type=int, default=0, choices=(0, 1))
    play.add_argument("--dealer", type=int, default=0, choices=(0, 1),
                      help="which seat deals the first hand")
    play.add_argument("--target", type=int, default=DEFAULT_TARGET)
    play.set_defaults(func=cmd_play)

    luck = sub.add_parser(
        "luck", help="attribute one game's cut luck to a player, round by round"
    )
    luck.add_argument("--p0", default="heuristic", help=f"seat 0 agent ({known})")
    luck.add_argument("--p1", default="heuristic", help=f"seat 1 agent ({known})")
    luck.add_argument("--player", type=int, default=0, choices=(0, 1),
                      help="whose side to report from")
    luck.add_argument("--seed", type=int, default=0)
    luck.add_argument("--target", type=int, default=DEFAULT_TARGET)
    luck.set_defaults(func=cmd_luck)

    bench = sub.add_parser("bench", help="measure scoring and simulation throughput")
    bench.add_argument("-n", "--games", type=int, default=200)
    bench.add_argument(
        "--agents", nargs="+", default=["random", "heuristic"], help="agents to time"
    )
    bench.set_defaults(func=cmd_bench)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
