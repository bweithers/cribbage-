# cribbage

A two-player cribbage simulator, written as the foundation for a search and
neural-net engine rather than as a game you play in a terminal.

Zero runtime dependencies — pure standard library. `pytest` is needed only to
run the tests.

```bash
python -m cribbage demo  --seed 3                             # a full annotated game
python -m cribbage match --p0 heuristic --p1 random -n 2000   # statistics with confidence intervals
python -m cribbage bench                                      # throughput
python -m pytest -q                                           # the test suite
```

## What's here

```
cribbage/
  cards.py       a card is an int in [0,52): rank = c >> 2, suit = c & 3
  scoring.py     table-driven scoring for the show and the play
  engine.py      CribbageState: legal_actions / apply_action / clone / determinize
  crib_table.py  expected crib value per lay-away, given a uniform opponent
  agents/        the agent interface, plus random and one-ply expected-value baselines
  arena.py       match running and statistics
  cli.py         demo / match / bench
```

Scope is deliberately narrow: **two players, 121 points, standard rules.** No
three-handed play, no five-card cribbage, no muggins.

## Design notes

**Cards are ints, not objects.** `rank = card >> 2`, `suit = card & 3`. The hot
paths stay allocation-free, `sorted()` orders by rank for free, and a card
doubles as an index into a 52-wide array — which is what a future net's feature
planes will want.

**Hand scoring is a dict lookup.** Fifteens, pairs and runs depend only on the
multiset of ranks, and there are just 6,175 legal five-card rank multisets, so
they are all enumerated at import. Only the flush and nobs — which need suits and
need to know which card was the starter — are computed per call. Result: about
1.5M hands per second, which is what makes the discard's exact 46-starter
enumeration below affordable.

**The engine is a state machine, not a callback loop.** `legal_actions()` /
`apply_action()` / `clone()`, in the style of OpenSpiel. A callback loop is easier
to write, but a tree search needs to fork a position and drive it itself, which
inverted control cannot offer.

**Chance is internal and forced transitions are fast-forwarded.** The deal and
the cut are resolved by the state's own RNG. `apply_action` runs through every
forced step — gos, count resets, the show, the next deal — so a caller only ever
sees a state that is a real decision or terminal, and `current_player` is never a
player with nothing to do.

**The discard is modelled sequentially** (pone, then dealer) although real
cribbage discards simultaneously. That is sound only because the dealer's
information state is identical no matter which two cards the pone laid away, which
is asserted directly in `tests/test_infoset.py`.

### The play phase

Most of the rules risk lives here, so it is worth stating what the loop gets
right:

- **A go is not a turn change.** When one player cannot play, the other keeps
  laying cards — scoring pairs and runs off their own sequence — until neither can.
- **Thirty-one pays 2 and does not also pay the go point.** Hitting 31 resets
  immediately, which is what stops the go branch from paying a second time.
- **The last card of the play pays 1**, unless that card made 31.
- **The next lead goes to the other player from whoever played last**, after both
  a 31 and a go.
- **Pairs and runs never reach across a reset.** A seven laid after a reset does
  not pair with the seven that ended the previous sub-round.
- **Runs use rank; fifteens and the count use value.** J-Q-K runs for three while
  counting thirty.

### Built for imperfect-information search

The eventual target is ISMCTS with a network, so three operations are already in
place, since retrofitting them would have meant rewriting the state machine:

- `clone()` — fork a position; deep, RNG state included.
- `information_state(player)` — what that player knows, and nothing more.
- `determinize(player, rng)` — sample a complete world consistent with that
  information state. The defining property, asserted in the tests, is that
  `world.information_state(p) == state.information_state(p)`.

Everything else a net needs — observation encoding, action indexing, self-play
recording, training — is deliberately **not here yet**.

## Agents

- `random` — uniform over legal actions. The floor.
- `heuristic` — one-ply expected value, and the baseline any future net has to beat.
  - *Discard*: the expected hand score is computed **exactly**, enumerating all 46
    possible starters for each of the 15 ways to keep four cards. The crib is then
    added as dealer and subtracted as pone, using `crib_table.py`.
  - *Play*: points scored now, minus the opponent's expected reply averaged over
    the cards they could still hold, plus the go point weighted by the
    hypergeometric chance they cannot answer. Familiar maxims — don't lead a five,
    don't take the count to 21 — are nowhere in the code; they fall out of the
    reply term.
- `greedy` — the same agent with the reply term switched off. Useful as a control.

### What the heuristic is not

One term inside it is exact: for a given four-card keep, `E[hand score]` over all
46 possible starters is computed by full enumeration, no sampling. The *policy*
built around that term is not exact, in four separate ways, and they are worth
knowing before anything is measured against it:

1. **The crib term assumes a uniform opponent.** `crib_table.py` weights every
   opponent lay-away equally. Real pones lay away defensively and real dealers lay
   away helpfully, so a stronger agent would want separate dealer and pone tables
   conditioned on opponent policy. Regenerate with
   `python scripts/build_crib_table.py`.
2. **The crib table ignores your own hand.** It is keyed on
   `(rank, rank, suited)` alone, but which six cards you hold changes what is left
   in the deck and therefore the crib distribution.
3. **It ignores pegging value entirely.** The discard maximizes hand plus crib and
   nothing else. Four fives peg badly and A-2-3-4 pegs well; this agent cannot see
   the difference. Serious discard analysis carries an expected-peg term.
4. **It maximizes points, not win probability.** `my_score` and `opp_score` are in
   the `InfoState` and the agent never reads them. At 118-115 you should play
   nothing like you do at 20-15.

The play side is 1-ply, averages the opponent's reply rather than assuming their
best one, and draws that reply from a pool that includes the crib and the undealt
deck — cards the opponent demonstrably cannot hold.

So it is a reasonable floor to measure against, not a ceiling. Gaps 3 and 4 in
particular are places a self-play net should be expected to *beat* it rather than
converge to it.

## Testing

Correctness is the whole point of this layer: any future net is trained against
these rules, so a scoring bug would not produce a slightly worse net, it would
produce a net that confidently plays a different game. Three independent lines of
attack:

1. **Known fixtures** — hand-derived values for the 29 hand, double and triple
   runs, the crib four-flush that scores nothing, nobs versus his heels, and every
   `score_play` rule including out-of-order and broken runs.
2. **Differential fuzzing** — a second, deliberately naive scorer in
   `tests/reference.py`, using different algorithms (brute-force subset
   enumeration for runs rather than multiplying rank multiplicities), fuzzed
   against the fast one over tens of thousands of random hands and play sequences.
3. **Exhaustive check** — `scripts/exhaustive_scoring_check.py` scores all
   2,598,960 five-card combinations against each of their five possible starters
   and asserts the classically impossible totals **19, 25, 26 and 27** never occur
   and that the maximum is 29. Also available as `pytest --runslow`.

The play phase is driven by scripted hands rather than random ones, so each
branch — go, thirty-one, last card, reset — is reached deliberately and checked
against an exact expected event sequence.

The end-to-end sanity check is external knowledge: the heuristic agent averages
**≈7.9 points per hand at the show**, which is where real well-played cribbage
sits. A rules bug the unit tests missed would move that number.

## Performance

From `python -m cribbage bench` on one core:

| | rate |
|---|---|
| `score_hand` | ~1.5 M/sec |
| `score_play` | ~0.7 M/sec |
| `apply_action` | ~80 K/sec |
| random self-play | ~410 games/sec |
| heuristic self-play | ~46 games/sec |

The engine is not the bottleneck for the heuristic agent — its 46-starter discard
enumeration is, at roughly 690 hand evaluations per discard. That is a deliberate
trade: an exact hand-EV term now, with the option to approximate it later. The
scoring and play hot paths are small isolated functions,
so they can be swapped for a vectorized or native implementation without touching
the rules.

## Next

The neural-net work is intentionally deferred. When it starts, the open questions
are, roughly in dependency order: observation encoding and what the net actually
sees; the action space (one masked head over 15 discard pairs plus 52 cards,
versus separate heads); where the net attaches to ISMCTS (policy prior, value
head, or both); how to handle the imperfect information honestly, including why
vanilla AlphaZero-style MCTS is unsound here and what ISMCTS and CFR do about it;
and whether to bootstrap from the heuristic discard policy before self-play —
bearing in mind it is pegging-blind and position-blind, so imitating it too
closely would inherit both.
