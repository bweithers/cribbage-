# cribbage

A two-player cribbage simulator, written as the foundation for a search and
neural-net engine rather than as a game you play in a terminal.

Zero runtime dependencies — pure standard library. `pytest` is needed only to
run the tests.

```bash
python -m cribbage play  --luck 75                            # play it yourself, cuts rigged
python -m cribbage demo  --seed 3                             # a full annotated game
python -m cribbage match --p0 heuristic --p1 random -n 2000   # statistics with confidence intervals
python -m cribbage bench                                      # throughput
python -m pytest -q                                           # the test suite
```

**Contents** — [what's here](#whats-here) · [design notes](#design-notes) ·
[agents](#agents) · [does position awareness matter?](#does-position-awareness-matter) ·
[where does the luck come from?](#where-does-the-luck-come-from) ·
[testing](#testing) · [performance](#performance) · [next](#next)

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
- `positional` — `heuristic` plus awareness of the score: a continuous stance that
  defends when ahead and reaches for variance when behind, and an endgame
  objective that maximizes the probability of going out this deal rather than
  expected points. Every channel is an independent flag, and with all of them off
  it reproduces `heuristic` decision-for-decision. See the measurements below —
  it is an experiment, not an improvement.

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

## Does position awareness matter?

`heuristic` maximizes points and never reads the score, which is plainly wrong
for a race to 121. `positional` adds two mechanisms — a continuous **stance**
that defends when ahead and reaches for variance when behind, and a sharp
**endgame** objective that maximizes the probability of going out this deal
rather than expected points. Each channel is a separate flag so it can be
ablated on its own.

Measuring this needs care. Cribbage deal luck is enormous: for agents this
similar, **96% or more of deal-pairs split** — the same cards flip the result
when the seats swap, so the deal decided the game, not the play. So `scripts/experiment_position.py` plays every deal sequence
twice with the seats reversed and computes intervals on per-pair scores rather
than per-game results. That cuts the interval roughly threefold for the same
compute, which is the difference between resolving these effects and not.

The control that everything rests on: `PositionalAgent` with every feature off
reproduces `HeuristicAgent` decision-for-decision, asserted in
`tests/test_positional.py`.

### Results

3,000 mirrored pairs (6,000 games) per row, against the `heuristic` baseline.
Seventeen comparisons at 95% confidence means roughly one is expected to clear
significance by chance, so the script says so in its own footer and any single
marginal winner gets re-run on fresh seeds before being believed:

| variant | win rate | 95% interval | verdict |
|---|---|---|---|
| endgame | 49.65% | [49.33, 49.97] | **significantly worse** |
| stance-variance | 50.12% | [49.85, 50.38] | no effect |
| stance-crib | 50.07% | [49.88, 50.25] | no effect |
| stance-peg | 49.93% | [49.80, 50.06] | no effect |
| crib-defense | 50.00% | [49.65, 50.35] | no effect |
| five-penalty | 49.92% | [49.80, 50.03] | no effect |
| finish | 50.00% | [50.00, 50.00] | never fires |
| all combined | 50.00% | [49.54, 50.46] | no effect |

**Position awareness, as implemented here, is worth approximately nothing.**

Four things are worth pulling out of that table.

**"Never throw points into their crib" is already priced in.** The folk rule
tests at exactly 50.00% (58 pairs swept each way), and an explicit extra penalty
on laying away a five is likewise null. This is not because defence does not
matter — it is because `crib_table.py` already charges the correct expected cost,
and 5-5 is already the most expensive lay-away in the table at 9.0 points.
Adding instinct on top of correct arithmetic just makes the arithmetic wrong,
and the sweep shows it degrading monotonically as the thumb presses harder:

| extra crib weight | win rate | verdict |
|---|---|---|
| 0.25 | 49.73% | no effect |
| 0.5 | 50.50% | no effect |
| 1.0 | 49.20% | **worse** |
| 2.0 | 47.97% | **worse** |
| 4.0 | 46.20% | **worse** |

Refusing to feed the crib means wrecking your own hand to do it, and past about
a point of thumb the trade stops being worth it.

**The one "significant" positive did not replicate.** Sweeping the variance gain
turned up `stance_variance=0.15` at +0.30% with an interval clearing 50%. Re-run
on fresh seeds at twice the sample size, the same idea came back at **+0.02%,
interval [49.87, 50.16]** — dead null. That is exactly the false positive you
expect from seventeen comparisons at 95% confidence, and it is why the
replication step is not optional.

**Some features are inert, and the win rate cannot tell you which.** `finish`
(play a card that wins outright) returned 50.00% with all 3,000 pairs splitting —
the signature of a feature that changed *zero* decisions, because greedy pegging
already picks that card. An earlier version of the whole positional agent scored
this way across the board; instrumenting how often each flag actually changes an
action, rather than trusting the null result, is what caught it.

**The endgame objective was actively harmful, and the reason was diagnosable.**
It thresholds on `to_go − expected pegging`, but pegging has a standard deviation
of about 2.2 points — comparable to the spread of a hand's score across starters
— so it optimized hard against a threshold that was itself badly uncertain.
Folding over the measured pegging distribution instead of its mean
(`endgame_smoothing`, on by default) is the principled repair. Tested head to
head on the same fresh seeds, 6,000 pairs each:

| endgame variant | win rate | 95% interval | verdict |
|---|---|---|---|
| smoothing on | 50.07% | [49.83, 50.31] | no effect |
| smoothing off | 49.70% | [49.47, 49.93] | **worse** |

So the diagnosis was right — the sharp threshold was the harm, and smoothing
removes it. It buys neutrality, not an edge.

### Why so little?

The baseline is already doing exact 46-starter enumeration with a correctly
signed crib term, and these features only perturb that objective slightly: they
change **1–3% of discards** and under 0.5% of pegging decisions. A small edge on
2% of decisions is a very small edge overall.

The implication is that the points are somewhere else. The baseline's real gap is
not that it ignores the score — it is that its discard **ignores pegging value
entirely**, which distorts every hand rather than 2% of them. That is the
experiment worth running next.

## Where does the luck come from?

Two identical deterministic agents remove skill from the picture entirely, which
makes a finished game a pure function of three independent inputs:

```
outcome = f(the cards dealt, the cuts turned, who dealt first)
```

There is no residual noise term, so the variance decomposes *exactly*.
`scripts/experiment_luck.py` runs a full factorial — K deal seeds × K cut seeds
× 2 dealers — and reads it two ways. This is what the engine's split RNG streams
(`seed` and `cut_seed`) are for: the deck always holds exactly 40 cards when the
starter is turned, so a given cut stream draws the same *index* every round
regardless of what was dealt. Holding one factor while re-randomizing another is
therefore proper common random numbers rather than an approximation, which
`tests/test_engine.py` asserts directly.

### Results

128 deals × 128 cuts × 2 dealers = **32,768 games**.

**Dealing first is worth more than any strategy difference measured anywhere else
in this repo.**

| | |
|---|---|
| first dealer wins | **56.17%** (95% CI 53.46–58.87, clustered by deal) |
| mean margin swing from dealing first | **+6.61 points** |

For scale: every position-awareness feature in the section above measured within
±0.35% of a coin flip. Winning the cut for first deal is worth six points of
expected margin before a card is played.

Why is it worth that much? Partly the obvious reason — games run about 9.1
rounds, so the first dealer deals 4.82 of them against 4.30, a **+0.52 deal**
edge. At the measured 6.0-point gap between dealing and not, that is only about
+3.1 points, so it accounts for roughly half the swing.

The rest looks like tempo. Splitting games by whether the first dealer actually
got more deals:

| | first dealer wins |
|---|---|
| they got one extra deal | 60.18% (n=3152) |
| both dealt equally | **53.65%** (n=2848) |

Even when both dealt the same number of times, dealing *first* is still worth
three and a half points of win rate — banking the crib earlier means leading
throughout and reaching 121 first. Read that second row as suggestive rather
than decisive, though: the number of rounds is itself influenced by how the game
goes, so conditioning on it is not a clean causal estimate.

#### Total influence: re-randomize one factor, how often does the winner change?

| re-randomize only… | winner changes |
|---|---|
| the cards dealt | 49.09% |
| who deals first | 37.45% |
| the cut cards | 31.96% |
| *everything* | *49.75%* |

The deal alone (49.09%) accounts for essentially all of the randomness available
(49.75%). Cribbage is close to being a game about who got the better cards.

These are computed exactly rather than by sampling pairs: for a fixed deal and
dealer the outcome is Bernoulli(p) across the cut axis, and two independent
draws disagree with probability `2p(1-p)`. The plug-in estimator is biased low
by a factor of `(1 - 1/n)`, since `p` is estimated from the same draws, so the
code applies the `n/(n-1)` correction.

#### Systematic advantage: variance of the final margin, by source

Total margin sd is 23.0 points.

| source | η² | ω² | F | p | effect sd |
|---|---|---|---|---|---|
| the cards dealt | 39.0% | 38.9% | 325.9 | <1e-12 | **14.42 pts** |
| deal × cut | 30.8% | 15.6% | 2.03 | <1e-12 | 9.10 pts |
| deal × who dealt first | 12.6% | 12.4% | 105.0 | <1e-12 | 8.16 pts |
| who dealt first | 2.1% | 2.1% | 2182.6 | <1e-12 | **4.67 pts** |
| the cut | 0.3% | 0.2% | 2.42 | <1e-12 | **0.96 pts** |
| cut × who dealt first | 0.1% | 0.0% | 0.94 | 0.66 | 0.00 pts |
| irreducible three-way | 15.2% | — | (error) | | |

η² flatters `deal` and `cut`, which have 127 degrees of freedom each, against
`who dealt first`, which has one — under the null, sum of squares grows with
degrees of freedom, so ω² subtracts exactly `df × MS_error` before dividing.
**The effect-sd column is the one to quote**: it is in points, so it answers
whether a factor *matters* rather than only whether it is *detectable*.

The cut is a clean example of why that distinction is needed. With 32,768 games
its main effect is overwhelmingly significant (F = 2.42, p < 1e-12) and utterly
negligible (0.96 points against a 23.0-point spread). Significance is a claim
about whether an effect is zero; it says nothing about size.

### The cut: large influence, near-zero *seed-level* advantage

Re-randomizing the cut **changes the winner 31.96% of the time** while its main
effect is worth **0.96 points**. Both are true. The starter is a *shared* card,
so what it is worth depends entirely on the hand it lands beside — which puts
almost all of its influence into the `deal × cut` interaction (9.10 points)
rather than into a main effect.

**But the ANOVA is the wrong instrument for "does a lucky cut favour the
dealer", and it is worth being explicit about why.** Its `cut` factor is a *seed
for an index sequence*, and the deck always holds 40 cards at the cut, so
whether index 7 is a good card depends on what was dealt. Averaged over 128
different deals, no cut seed can persistently favour a seat — the near-zero main
effect is close to guaranteed by the design, not a discovery. The same objection
applies to `cut × who dealt first` (F = 0.94, p = 0.66).

The question has to be asked about the *card*, not the seed. The dealer scores
the starter against two holdings — their hand and their crib — where the pone
scores it against one, so a good starter should be worth more to the dealer.
Points per deal, by the rank turned (~940 deals per row):

| starter | dealer | pone | gap |
|---|---|---|---|
| **5** | 19.36 | 11.98 | **7.38** |
| **J** | 17.50 | 10.10 | **7.39** |
| 8 | 15.92 | 9.85 | 6.07 |
| 7 | 15.91 | 10.04 | 5.87 |
| 2 | 15.49 | 9.75 | 5.74 |
| 6 | 15.87 | 10.39 | 5.48 |
| K | 14.96 | 9.46 | 5.50 |
| 3 | 15.25 | 10.06 | 5.19 |

So a lucky cut **does** favour the dealer, by up to 2.2 points more than an
unlucky one. Turning a five is worth 3.7 extra points to the dealer and 1.9 to
the pone; a jack pays the dealer his heels outright. The seed-level ANOVA simply
cannot see this, and reading its near-zero `cut` row as "the cut is fair" would
have been wrong.

The deal is the mirror image: it is the one input that is **not** shared, so it
is where the persistent, seat-specific unfairness lives.

### Scoring one game's cut luck

`cribbage/luck.py` answers a narrower and more practical question: *given this
game's log, how lucky were the cuts for me?* — as one number.

The counterfactual set is small and completely known. Once the discards are
settled, both holdings, the crib and the forty undealt cards are all fixed, and
the only thing that could have gone differently is which of those forty came up.
So score all forty:

```
cut luck  =  (what the real starter gave you)  −  (what an average starter gives you)
```

Subtracting the expectation does two jobs at once. It removes the dealer's
systematic crib-and-heels edge, which is position rather than luck. And it nets
out cards that help both players — if a five drops and you both use it, it
contributes nothing. Because all forty candidates are enumerated rather than
sampled, the standard deviation comes out exactly, so "how many sd was that" is
available with no distributional assumption. Round variances add, giving a
game-level z-score.

```
$ python -m cribbage luck --seed 3

   round   cut   show    exp   gross     net     sd      z
       1    JD     17   18.5    -1.5    -3.2    5.4  -0.60
       2    5C      8    5.9    +2.1    +4.2    4.5  +0.93
       3    5H     15    8.8    +6.2    +8.0    4.5  +1.79
       4    5S      6    7.3    -1.3    -3.2    2.3  -1.39
       …
```

Note rounds 2, 3 and 4 all turn a five, and round 4's is worth **−3.2** — it
helped the opponent more. A card is not lucky in itself, only against the hand
it lands beside.

**Is the number real?** Over 2,500 games it is centred on zero (+0.21, sd 11.8)
as the zero-sum construction requires, and its z-score has **sd 0.994** — an
independent check that the per-round variances and their summation are right. It
correlates +0.58 with final margin, and sorts outcomes hard:

| cut luck | net points | win rate |
|---|---|---|
| worst 25% | −14.7 | **18.7%** |
| 2nd | −3.9 | 39.7% |
| 3rd | +4.1 | 57.6% |
| best 25% | +15.3 | **81.4%** |

**What it deliberately does not say.** The 29 hand — three fives and the
matching jack, case five cut — scores only **+5.3 net, z = +0.91**. That is
correct: three fives and a jack already average 16.9 points against a *random*
starter, so being dealt it is the luck and the cut merely finished it. The
measure separates the two on purpose.

Scope: this is the cut as realised at the show — hands, crib, his heels, nobs.
Pegging is excluded; the starter barely touches it and could not be attributed
cleanly. Rounds cut short by someone winning before the show are excluded and
counted separately.

### Playing a rigged game

`python -m cribbage play --luck 75` deals you a real game against the heuristic
agent and turns **every** starter at the 75th percentile of what it could have
been for you, so you can feel what a given grade of cut luck is actually like.

This is exact, not a search, and the reason is a nice bit of timing: **the cut
happens after both discards.** A seed has to be fixed before anyone has thrown
anything, so it cannot know what a card will be worth — the same five is a gift
beside one holding and a blank beside another. By the time the starter is
turned, the holdings and the crib are settled and all forty remaining cards can
be scored exactly. `PercentileCut` ranks them and turns the one you asked for.

The interesting thing it exposes is how per-round luck **compounds**. Every cut
at the 75th percentile is not a 75th-percentile game:

| every cut at | game z | net points |
|---|---|---|
| 0th percentile | −4.48 | −48.6 |
| 25th | −1.82 | −22.4 |
| 50th | +0.06 | +0.7 |
| 75th | **+1.82** | **+22.2** |
| 100th | +4.54 | +48.9 |

Per-round edges add, while their standard deviations only add in quadrature, so
a steady 75th-percentile cut lands near z = +1.8 for the game — worth about 22
points, roughly a fifth of the board. Playing a few at 25 and a few at 75 is a
faster way to calibrate what "the cards ran against me" is worth than any table.

### The three questions, answered

1. **Did you get the first deal?** A large systematic edge — 56/44, +6.6 points
   of margin, effect sd 4.67 points — but only ~2% of game-to-game variance,
   because it is one binary switch against the enormous variety of card
   sequences. Big effect, small variance share; the two are not the same thing.
2. **Were you dealt better cards?** The dominant source of both: effect sd
   **14.42 points**, ~39% of margin variance on its own, and it drives most of
   the rest through its interactions. Re-randomizing the deal alone reproduces
   almost all the randomness in the game.
3. **Did you get luckier cuts?** It flips a third of games, but almost entirely
   through *which hand it joins* rather than through favouring a player. Its
   own systematic contribution is about a point of margin. It is not neutral,
   though: a good starter is worth roughly two points more to the dealer than to
   the pone, which the card-level table above shows and the seed-level ANOVA
   cannot.

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
