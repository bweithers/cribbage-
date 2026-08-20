"""Deliberately naive scorers, written to disagree with the fast ones.

These exist only to be differentially fuzzed against :mod:`cribbage.scoring`.
They are slow, allocate freely, and use different algorithms on purpose -- in
particular the run detection here brute-forces subsets rather than multiplying
rank multiplicities, so a bug in the fast table cannot hide behind a matching
bug here.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence


def _value(card: int) -> int:
    return min((card >> 2) + 1, 10)


def naive_score_hand(hand: Sequence[int], starter: int, is_crib: bool = False) -> int:
    cards = list(hand) + [starter]
    ranks = [c >> 2 for c in cards]
    suits = [c & 3 for c in cards]
    values = [_value(c) for c in cards]
    total = 0

    # Fifteens: brute force every subset.
    for size in range(1, 6):
        for combo in combinations(range(5), size):
            if sum(values[i] for i in combo) == 15:
                total += 2

    # Pairs: brute force every unordered pair.
    for i, j in combinations(range(5), 2):
        if ranks[i] == ranks[j]:
            total += 2

    # Runs: find the longest subset length that forms a run, count how many
    # subsets of that length do, and score length * count.  Shorter runs nested
    # inside a longer one do not score, which is why only the longest length is
    # considered.
    for size in (5, 4, 3):
        found = 0
        for combo in combinations(range(5), size):
            sub = sorted(ranks[i] for i in combo)
            if len(set(sub)) == size and sub[-1] - sub[0] == size - 1:
                found += 1
        if found:
            total += size * found
            break

    # Flush.
    if len(set(suits[:4])) == 1:
        if suits[4] == suits[0]:
            total += 5
        elif not is_crib:
            total += 4

    # Nobs.
    for i in range(4):
        if ranks[i] == 10 and suits[i] == suits[4]:
            total += 1

    return total


def naive_score_play(seq: Sequence[int]) -> int:
    if not seq:
        return 0
    ranks = [c >> 2 for c in seq]
    total = 0

    count = sum(_value(c) for c in seq)
    if count == 15:
        total += 2
    if count == 31:
        total += 2

    # Pairs: count how many trailing cards share the final rank.
    matching = 0
    for r in reversed(ranks):
        if r == ranks[-1]:
            matching += 1
        else:
            break
    if matching >= 2:
        total += matching * (matching - 1)

    # Runs: longest qualifying suffix, checked by sorting rather than by span.
    for size in range(len(seq), 2, -1):
        tail = sorted(ranks[-size:])
        if len(set(tail)) == size and tail == list(range(tail[0], tail[0] + size)):
            total += size
            break

    return total
