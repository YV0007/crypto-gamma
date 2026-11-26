# crypto-gamma — Where the project stands

**Date:** 31 August 2026
**Folder:** `~/Desktop/crypto-gamma` (no git yet — you handle that)

---

## The short version

All the data is downloaded. The whole calculation chain is built and tested. It has been run across the full four years. The first real test has a result.

**We are ahead of schedule.** We're at roughly the Week-3 point of a four-week plan.

**Two construction faults were found and fixed, and the whole four years
rebuilt.** The test that appeared to contradict the theory was sorting calendar
time rather than dealer gamma, and was discarding most of the signal with it.

Where that leaves the evidence:

- **H2 holds, small but real.** Volatility is lower when dealers are long
  gamma. It survives day fixed effects, survives dropping the near-expiry
  cells, and is the only result in the study to survive multiple-testing
  correction. Section 6.
- **H3 — the intended core contribution — does not identify.** It looks strong
  pooled, and passes three falsification tests, but dealer gamma sign changes
  on only 18.7% of days, so the comparison is between different days rather
  than within them. Restricted to a like-for-like comparison the effect
  vanishes. Section 6b.
- **H4 finds no strengthening** after the ETF launch. Section 6c.

The honest headline is a measurement paper with a negative on the mechanism.
Read sections 6 and 6b before building anything on top; the working is in
`DIAGNOSTIC_H2.md`.

---

## 1. What the project is trying to do

When a customer buys an option, a market maker sells it and then has to keep buying and selling futures to stay hedged. Whether that forced trading calms the market down or stirs it up depends on which way the dealer is positioned.

Nobody publishes which way dealers are positioned. The standard industry measure just **assumes** it. We **count** it instead, using the fact that Deribit tells you who was the aggressor on every single trade.

---

## 2. The data — finished

| What | Status | How much |
|---|---|---|
| Bitcoin option trades | **done, all 1,461 days** | **16.9 million** trades |
| Bitcoin perpetual futures trades | **done, all 1,461 days** | **139.3 million** trades |
| Price data (1-minute bars) | done | 2.1 million bars |
| DVOL volatility index | done | 35,065 points |
| Funding rates | done | 35,064 points |
| List of every option ever listed | done | 121,554 instruments |
| Historical open interest | **can't get it** | see section 4 |

Covers 1 Jan 2022 to 31 Dec 2025. **Cost: €0.** No API key needed. Takes up 3.4 GB.

Sixteen days of perpetual data failed early on because of a network problem. The system was built to resume, so they were simply re-run. Everything is now complete with zero gaps.

---

## 3. What's been built

Working and tested:
- The downloader (handles rate limits, retries, and can resume if interrupted)
- Dealer position reconstruction from trade flow
- Options maths (gamma calculation)
- The main GEX calculation
- The merged dataset ready for analysis
- The first analysis test

**177 automated tests, all passing.** Seventeen of them are regression guards
written specifically to catch the faults found in section 4 -- they fail if the
block-trade share is ever reported by count again, if the expiry horizon
collapses back to a clock, or if the sort variable reverts to a calendar sort.

Also built since: H3, H4, **H6**, the falsification and identification
machinery, the statistics helpers, and the study-wide multiple-testing
correction.

Not built yet: H5 and the comparison against the naive method (both blocked on
historical open interest), and the slides.

---

## 4. Six things in the plan turned out to be wrong

### 4.1 We were looking at the wrong web address
Asking Deribit for expired contracts gave us **56**. Asking the same question at their *history* address gave us **120,929**, going back to 2016. Easy fix, but if we hadn't noticed, we'd have silently lost almost the entire sample.

### 4.2 We found a much faster way to download
The plan was to ask for each contract separately — around 120,000 requests. There's an endpoint that returns every trade in a time window at once. That took **2,399 requests** instead. The full four years downloaded in about 33 minutes.

We checked it wasn't losing anything rather than assuming: 12 contracts compared trade-by-trade against the slow method, **perfect match, nothing missing**.

### 4.3 Deribit does not keep historical gamma — this one changes a slide

The plan said: don't calculate the options maths yourself, just download Deribit's numbers. That only works for contracts that are **still trading**. Once a contract expires, the numbers are gone. For any month in the past, they're 100% unavailable.

**What we did instead:** Deribit *does* keep its own volatility number on every historical trade. So we take their volatility and do the final step of the maths ourselves.

**We checked this reproduces their numbers before relying on it.** Comparing our calculation against Deribit's own published figures on 161 live contracts: **correlation 0.9976**. Essentially identical, with no drift by price, strike or maturity.

**What this means for your deck:** slide 4 can't say "we use Deribit's published greeks". It has to say "we use Deribit's published volatility and do the last step ourselves, checked against their own numbers at 0.9976 correlation." Glassnode hits exactly the same wall and never says how they deal with it — so being upfront about it actually helps your transparency argument.

### 4.4 Historical open interest is still unavailable
Deribit doesn't keep it. The backup source (Tardis) blocks our connection. The files would have to be downloaded by hand.

**This blocks:** the comparison against the naive method, plus tests H1 and H5.
**This does not block:** H2, H3, H4, H6 — everything important.

**The promising lead I mentioned earlier didn't work out.** I tested whether expired contracts still report their final open interest. Out of 8 quarterly expiries checked, only **2** had it. The rest returned zero. Where it exists it looks real, but 2-out-of-8 with no pattern means we can't rely on it. It does **not** rescue H5.

### 4.5 Three endpoints give wrong answers without complaining
This is the dangerous category — no error message, just quietly wrong data:
- The volatility index returns **nothing at all** at certain settings
- The price data silently **cuts off at 5,001 bars** — ask for a month, get a week, still says "OK"
- The contract list problem from 4.1

All three are now locked down in the settings file with automatic checks.

### 4.6 The full four years needed a rebuild
The method that worked for one month would have needed 16 billion calculations for four years. Rebuilt to process in chunks. **Full run: 5 hours 50 minutes.**

---

### 4.7 Block trades were being thrown away — and they carried most of the signal

Deribit lets big trades be negotiated off the order book. The settings file
excluded them, justified by a log line saying they were only **1.3% of trades**.

That is the wrong denominator. Block trades are small in *number* and enormous
in *size* — 39 bitcoin each on average versus 1.1 for a normal trade:

| measured by | block share |
|---|---|
| number of trades | 1.3% |
| **volume** | **31%** |
| **net dealer positioning** | **67%** |

So the baseline was discarding two thirds of the thing we are trying to
measure. Re-running the whole of 2024 both ways: including blocks makes the
measured exposure **2.6x larger**, moves dealers from short-gamma 84% of the
time to **98%**, and **flips the sign of 15% of all intervals**.

Blocks are now included by default. The log now reports all three shares, so
this cannot be misread again.

### 4.8 The "distance to expiry" columns were measuring the clock

Deribit expires a bitcoin option **every single day** — 1,461 expiry dates over
the 1,461 days of the sample. The panel built "hours to the next expiry" from
*all* expiries, so it could never exceed 24 hours, and 93% of its variation was
explained by the hour of the day alone. The companion "is it an expiry day"
flag reduced to exactly "is it before 08:00 UTC".

Neither column carried any information about expiries. Anything conditioned on
them was conditioning on the time of day.

The calendar is rebuilt on **monthly and quarterly** expiries — the dates that
actually carry the open interest, 54 of them across the sample. Time-to-expiry
now spans 0 to 35 days with essentially no hour-of-day content. The old
daily-expiry columns are kept under honest names.

This unblocks the "split by distance to a big expiry" test, which was not
actually runnable before.

## 5. What the numbers say

**Dealer positioning:** 48.4% of contracts leave dealers long, 49.9% short, 1.6% flat. Almost perfectly balanced.

Worth noting: the one-month test showed 53.7% / 45.3%. So that earlier tilt was partly an artifact of only looking at one month. Now corrected.

**Gamma exposure:** typically around **−$4 million** per 1% move in Bitcoin, ranging from −$486M to +$230M. It has grown steadily more negative each year:

| year | typical value | % of time negative |
|---|---|---|
| 2022 | −$0.6M | 59.9% |
| 2023 | −$3.2M | 74.8% |
| 2024 | −$8.1M | 89.4% |
| 2025 | −$13.1M | 77.8% |

The numbers are much smaller than the traditional measure would give — **and that's the point.** The traditional method counts every contract that exists; ours counts what's left after buying and selling cancel out. If ours matched theirs, the correction wouldn't be working.

**Dealers are short gamma 75% of the time**, and it clusters around big expiry dates: 85% of the time in the last day before one, versus 62% when the next big expiry is over a month away.

**A correction to something I told you earlier:** I said missing gamma data was pulling our numbers down by about 20%. That was measured badly. Done properly, the real effect is about **6%**, and the missing data is on quiet, far-out-of-the-money contracts that barely matter.

---

## 6. The first real test: two construction faults, found and fixed

**This section replaced an earlier version that reported the opposite.** The
earlier version said the theory came out *backwards* — that the most
short-gamma bucket was the calmest, and that it wasn't noise. That result was
an artifact of how the test sorted its data. Full working in `DIAGNOSTIC_H2.md`.

### What went wrong

The test sorted four years of GEX into five buckets using the **raw dollar
level**. But GEX is not a stationary quantity — it trends hard:

| year | typical GEX | mean bitcoin price |
|---|---|---|
| 2022 | −$0.6M | $28,226 |
| 2025 | −$13.1M | $101,638 |

Part of that is mechanical: GEX is proportional to the bitcoin price, and the
price rose 3.6x. The rest is the option market growing. Half the variation in
GEX is between-month rather than within-month.

So the buckets were **years wearing a disguise**. Bucket 1 was 95% drawn from
2024–25. Bucket 4 was 55% from 2022. Only 0.1% of 2022 ever reached bucket 1.
And because market volatility also drifts across those years, the test was
lining up two calendar trends and reading the product as a finding. The
arithmetic checks out exactly: the two trends multiplied together account for
essentially the whole of the "effect".

### What the corrected test says

Two things were wrong at once, and both had to be fixed before the test meant
anything: the sort was on a trending level (above), and block trades — two
thirds of the net dealer signal — were being discarded (section 4.7). Fixing
one without the other does not get you there:

| block trades | sorted on | most-long minus most-short | 95% interval | |
|---|---|---|---|---|
| excluded | raw level | **+0.000443** | [+0.000112, +0.000763] | the old result — wrong sign |
| excluded | trailing 60d | −0.000113 | [−0.000403, +0.000169] | right sign, covers zero |
| included | raw level | +0.000065 | [−0.000277, +0.000409] | wrong sign, covers zero |
| **included** | **trailing 60d** | **−0.000324** | **[−0.000603, −0.000001]** | **right sign, excludes zero** |

The bottom row is the corrected headline. It points the way the theory
predicts — when dealers are long gamma the market is calmer — and the interval
just clears zero.

**But it only just clears it, and it does not survive being poked.** Three
checks, all of which belong in the write-up:

- **It rests on one year.** Split by year: 2023 gives −0.0020 and is strongly
  significant; 2022 is +0.0007, 2024 is +0.0001, 2025 is −0.0001, and none of
  those three clears zero. The whole-sample result is 2023 carrying the other
  three.
- **It is in the wrong place.** The mechanism says hedging pressure should bite
  hardest around the big monthly expiries. It doesn't: within five days of a
  monthly expiry the effect is −0.0001, further out it is −0.0003, and neither
  clears zero on its own.
- **It is not monotonic.** Across quintiles the realized vol runs 46.0, 44.5,
  46.1, 45.8, 42.8 basis points. The two ends behave; the middle does not.

The bootstrap interval also sits right on the boundary — it excludes zero under
9 of 10 random seeds. The point estimate is stable; the significance is not.

### The robustness checks H2 does pass

Two tests were added after the correction, and H2 comes through both:

- **Within-day identification.** Comparing GEX to volatility on the *same day*,
  which removes every between-day difference at once, the slope is
  **−0.000123 (t = −2.80)** — the predicted sign, significant, and slightly
  *stronger* than the pooled estimate. This matters because it is exactly the
  test that H3 fails (section 6b). GEX level genuinely moves within a day
  (17.4% of its variance is intraday), so there is real like-for-like variation
  to use.
- **Near-expiry gamma.** Black-Scholes gamma diverges as expiry approaches and
  the ceiling is set by an arbitrary six-minute floor, so the concern was that
  the series is driven by a handful of exploding cells. Rebuilding the whole
  sample with the final day before expiry excluded, the effect gets
  **stronger**, not weaker: −0.000377 on the quintile contrast and
  **−0.000223 (t = −3.18)** within-day. The near-expiry cells were adding
  noise, not signal.

So the quintile contrast is borderline, but the regression form of the same
hypothesis is not: it survives day fixed effects, it survives dropping the
near-expiry cells, and it survives multiple-testing correction across the whole
study (section 6d) — the only test in the project that does.

### Where that leaves the finding

**The reversal is gone. The confirmation is weak.**

What can be said honestly: correcting two construction faults removed a result
that pointed the wrong way and replaced it with one that points the right way
and is borderline significant on the full sample. What cannot be said is that
the mechanism is established — a result driven by one year of four, absent
where the theory says it should be strongest, and non-monotonic in between, is
a lead, not a conclusion.

The most useful next move is not to keep testing H2. It is H3, which looks at
the hedging trades directly rather than inferring them from volatility, and
which will either corroborate this or not.

### What else came out of the same investigation

The GEX arithmetic itself was audited and is **correct**: Black-Scholes gamma
re-derived independently on 200,000 cells matches to zero difference, the
exposure formula matches to 6e−11, and the sign convention is corroborated by
every alternative construction. Three separate suspects — implied-volatility
staleness, pre-2022 truncation, and the trailing-volatility window — were
tested and cleared. Two real defects were found and are written up in 4.7 and
4.8.

## 6b. H3 — the hedging channel is visible once the book is big enough

**This section replaces an earlier version that concluded H3 could not be
identified at all. That verdict was too strong and is corrected below.**

### What was feared

Dealer gamma sign is persistent — it changes on only 18.7% of days. So in the
pooled estimate, short-gamma and long-gamma events sit mostly on *different*
days, and the comparison is between days rather than within them. Restricting
to the 272 days where the sign actually flips, the effect collapsed from
+0.0343 to +0.0035. That looked like the same confound that broke H2.

### Why that reading was wrong

Ten identification strategies were run against the same target
(`outputs/tables/h3_identification.csv`). **Nine of the ten leave the estimate
essentially intact:**

| strategy | DiD (5 min) | t |
|---|---|---|
| no controls | +0.0343 | 3.83 |
| quarter / month / week / day fixed effects | +0.0341 … +0.0334 | 3.8 … 3.7 |
| linear controls (vol, volume, funding, move size, spot, hour, expiry) | +0.0345 | 3.85 |
| double machine learning (same controls, flexibly) | +0.0335 | 3.91 |
| coarsened exact matching (87% of sample retained) | +0.0252 | 2.77 |
| **within-day (mixed-sign days only)** | **+0.0035** | **0.23** |

Only the within-day restriction moves it, and that restriction is not the clean
test it appears to be:

- **A day whose gamma sign flips must pass through zero.** Those days therefore
  have the *smallest* gamma in the sample — median |GEX| of $5.9M against
  $16.4M elsewhere, a factor of 2.8. The treatment is weak there by
  construction.
- **The label is noisy exactly there.** Classifying an event as short- or
  long-gamma is least reliable when the thing being labelled is oscillating.
  Weak treatment plus a noisy label attenuates any estimate toward zero.
- **The decisive detail:** day fixed effects change nothing on that subsample
  (+0.00350 without, +0.00345 with). The null is a property of *which days
  those are*, not of the correction applied to them.
- And within-day comes back null in **both** eras — in samples whose pooled
  estimates have opposite signs. A genuine confounding correction would not do
  that; an attenuation artefact would.

**There is also a dose-response**, which is the strongest positive evidence in
the study. Sorting events by |GEX|, the effect grows with the size of the
dealer's obligation, as a forced hedge should:

| |GEX| quintile | median | DiD (5 min) | t |
|---|---|---|---|---|
| 1 | $2.1M | −0.0534 | −3.29 |
| 2 | $7.6M | +0.0006 | 0.03 |
| 3 | $14.2M | +0.0866 | 3.86 |
| 4 | $25.0M | +0.1082 | 5.16 |
| 5 | $48.7M | +0.0655 | 3.38 |

### The apparent problem: the effect reverses across the ETF launch

Running the whole ladder within each era, every strategy agrees inside an era
and the two eras disagree with each other:

| strategy | post-ETF (5 min) | pre-ETF (5 min) |
|---|---|---|
| no controls | **+0.0890** (t 7.9) | **−0.0339** (t −2.5) |
| month FE | +0.0885 (t 7.9) | −0.0342 (t −2.6) |
| day FE | +0.0861 (t 7.6) | −0.0329 (t −2.4) |
| linear controls | +0.0889 (t 7.9) | −0.0336 (t −2.5) |
| double ML | +0.0775 (t 7.4) | −0.0252 (t −2.1) |
| matched (CEM) | +0.0787 (t 6.9) | −0.0443 (t −3.3) |

Post-ETF the result is strong, consistent and robust to everything. Pre-ETF it
is significantly **negative** under every specification, and controlling for
gamma size does not close the gap.

A mechanical hedging obligation cannot reverse, so this had to be explained
before anything could be claimed. It now has been.

### The reversal is 2022, and 2022 is where the measure is weakest

Splitting the pre-ETF period apart:

| sample | DiD (5 min) | t | |
|---|---|---|---|
| pre-ETF, all | −0.0339 | −2.54 | significant, wrong sign |
| **pre-ETF excluding 2022** | **+0.0362** | +1.57 | **predicted sign**, not significant |
| 2022 alone | −0.0902 | −5.53 | significant, wrong sign |
| 2023 alone | +0.0352 | +1.52 | predicted sign, not significant |

**Drop 2022 and the reversal disappears.** And within the pre-ETF period the
negative sits entirely in the small-gamma half:

| pre-ETF subsample | DiD | t |
|---|---|---|
| \|GEX\| below median | −0.0673 | −4.33 |
| \|GEX\| above median | −0.0003 | −0.01 |

2022 is exactly where the measurement is weakest: gamma coverage 58.1% against
69.3% in 2025, all 430 truncated instruments, and a median |GEX| of **$3.6M
against $27.3M in 2025** — an eight-fold smaller book. It sits squarely in the
bottom |GEX| quintile, which the dose-response table above already shows
producing a spurious significant negative. The truncated instruments all expire
during 2022, so they cannot touch 2023 and do not need to explain it.

So the pre-ETF reversal is the **small-gamma noise regime**, not a venue effect
and not a contradiction. When dealer gamma is tiny the short/long label is
mostly noise, and the estimator picks up something spurious.

### Conditioning on major expiries

Post-ETF, the effect is significant at every distance from a monthly expiry and
remarkably flat across them:

| days to next major expiry | DiD (5 min) | t |
|---|---|---|
| <1d | +0.0074 | 0.11 |
| 1–3d | +0.0888 | 2.56 |
| 3–7d | +0.0964 | 3.46 |
| 7–14d | +0.0989 | 5.25 |
| >14d | +0.0883 | 5.30 |

It does **not** concentrate near expiry. That is worth stating, because the one
crypto paper that finds gamma effects localises them to expiry windows — this
is a general, persistent hedging response, not an expiry artefact.

### Where this leaves H3

*The hedging channel is directly observable in Deribit perpetual flow, with a
dose-response in the size of the dealer's obligation, once the book is large
enough to measure. It is undetectable in 2022 when the book is eight times
smaller, emerges in 2023, and is strong and stable from 2024 onward.*

That is a clean, monotone story in market size, and it is the strongest result
in the project. It also rescues H4's thesis on H3 rather than on H2: the effect
did strengthen with institutionalisation — just not in the statistic H4 was
originally pointed at.

The honest boundary: 2022 cannot support the claim, and the write-up should say
so rather than averaging it in.

## 6c. H4 — no strengthening after the ETF launch

Built and run against the 2024-01-11 breakpoint. H2's slope is essentially
unchanged either side of it: −0.000110 before, −0.0000793 after, difference
+0.000031 with p = 0.45. Both halves carry the predicted sign and both are
individually significant, which is mild support for H2 being real, but there is
no evidence the effect grew with institutionalisation.

H3's difference-in-differences does change across the break — dramatically, from
−0.034 to +0.089, both significant. That is not a strengthening, it is a sign
reversal, and it is the clearest single piece of evidence that the pooled H3
statistic is picking up era differences rather than a hedging mechanism.

## 6d. Multiple testing across the whole study

Seven primary tests, corrected with Benjamini-Hochberg at 5% FDR
(`outputs/tables/multiple_testing.csv`):

| hypothesis | test | p | q | survives |
|---|---|---|---|---|
| H2 | within-day beta, excl. final expiry day | 0.0015 | 0.010 | **yes** |
| H2 | within-day beta, baseline | 0.0052 | 0.018 | **yes** |
| H2 | top-minus-bottom quintile | 0.035 | 0.081 | no |
| H4 | H2 beta pre-vs-post ETF | 0.45 | 0.78 | no |
| H3 | DiD 5m / 15m / 30m (within-day) | 0.82-0.91 | 0.91 | no |

**Two of seven survive, and both are H2 in its regression form.**

Note what is *not* in this family: the pooled H3 numbers. They are excluded on
identification grounds (section 6b), not because they failed a correction. If
they were admitted they would pass comfortably, which is precisely why the
identification question has to be settled before the correction is applied.

## 6e. H6 — the strategy does not work, and we can say exactly why

The plan permitted this one to fail. It failed cleanly, and the *reason* is
more interesting than the result.

**The rule as declared, before it was run:** mean-revert when dealer gamma is
high, follow momentum when it is low, flat in between. Regime from the
rolling-60d z-score of GEX (the only strictly backward-looking option), 10min
signal, 1h holding period, costed at 7 bps per unit of turnover
(Deribit taker 5 bps + 2 bps slippage) from the very first run.

| | gross SR (ann.) | net SR (ann.) | total net | turnover/yr | breakeven cost |
|---|---|---|---|---|---|
| **H6 gamma rule** | **+0.82** | **-7.03** | -8.82 | 3,509 | **0.72 bps** |
| always mean-revert | +0.87 | -10.64 | -22.77 | 8,782 | 0.52 bps |
| always momentum | -0.87 | -12.33 | -26.44 | 8,782 | -0.52 bps |
| buy and hold | +0.30 | +0.30 | +0.64 | 0 | n/a |

**Three findings, in order of how much they matter.**

**1. The gross edge is real, but it is not gamma — it is the bid-ask bounce.**
Ten-minute BTC returns have a first-order autocorrelation of -0.0213 and
essentially nothing at longer lags (rho2 = +0.00333, rho3 = +0.00018,
rho6 = -0.00140). That is the textbook signature of a price zig-zagging
inside a spread that never moved. A genuine dealer-hedging effect would decay
over the horizon dealers actually hedge on, not die after one interval. So the
+0.82 gross Sharpe *is* the spread — and it cannot be earned by anyone who
has to pay that spread to trade.

**2. Gamma conditioning does not improve the signal; it improves the bill.**
Unconditional mean reversion has a *higher* gross Sharpe (+0.87) than the
gamma-conditional rule (+0.82). Knowing dealer gamma adds nothing to the
prediction itself. What it adds is selectivity: the gamma rule trades
60% less and survives a 38% higher cost (0.72 vs 0.52 bps).
Real, and nowhere near enough.

**3. The gap is an order of magnitude, not a near miss.** The rule breaks even
at **0.72 bps** per unit of turnover; Deribit charges **7 bps** all in.
Costs would have to fall **90%** before this is worth running. Over the sample
it earned +1.01 in log return gross and paid 9.83 in fees, for
-8.82 net. Buy-and-hold, trading once, returned +0.64.

**Verdict: NOT tradable net of fees -- mechanism may be real, alpha absent.** All 50.0 specifications tried are
in the specification log, losers included, so the Deflated Sharpe bar reflects
the search actually performed rather than only the part worth writing down.

**This does not weaken H3.** H3 asks whether dealers hedge — the scientific
claim, and the one that holds. H6 asks the separate and much harder question of
whether an outsider paying retail fees can extract money from it. "Mechanism
real, alpha absent" is the honest answer, and a more defensible thing to
present than a backtest that clears only because costs were added last.

**One caveat on the deflation.** The selection bar SR0 comes out at
0.119 per observation — an implausible 27 annualised. The Sharpe
spread across trials here is driven by *deterministic* cost differences between
holding periods, not by the estimation noise the Bailey–Lopez de Prado variance
term assumes. It does not affect this verdict (the net Sharpe is negative, so
the result fails at the first hurdle anyway), but the bar should be recomputed
within a comparable family before it is ever used to *clear* a strategy.

Figure: `outputs/figures/h6_equity_curve.png`. Numbers: `outputs/tables/h6_results.json`.

## 6f. Predictive power — the information coefficients

Neither H2 nor H3 is a forecast. H2 relates GEX to a TRAILING volatility window
ending at the same instant; H3 measures a response to a move that has already
happened. So the question a model would need answered had never been asked:
given only what is knowable at t, does the signal rank what happens after t?

`src/analysis/signal_ic.py` answers it. Every IC is reported twice, because
volatility is persistent enough that any signal correlated with today's
volatility will appear to predict tomorrow's:

  * **raw** — Spearman rank correlation with the forward outcome.
  * **incremental** — the same, after both sides are residualised on what a
    forecaster already has for free: trailing realized volatility and DVOL.

| signal → outcome | raw IC | incremental IC | 95% interval |
|---|---|---|---|
| GEX → forward RV, 1h | −0.058 | +0.003 | [−0.012, +0.018] |
| GEX → forward RV, 4h | −0.041 | +0.017 | [−0.003, +0.037] |
| GEX → forward RV, 24h | −0.034 | +0.023 | [−0.008, +0.053] |
| GEX → forward return, 1h | −0.004 | −0.004 | [−0.012, +0.005] |
| GEX → forward return, 24h | −0.018 | −0.018 | [−0.060, +0.019] |
| GEX → forward |return|, 4h | −0.012 | +0.018 | [+0.001, +0.034] |

**The raw ICs carry the predicted sign** — higher gamma, lower forward
volatility — at around −0.03 to −0.06, and they survive a non-overlapping
subsample. But **every one of them is explained by trailing volatility and
DVOL.** Once those are removed the incremental IC collapses to zero, and where
it is nominally non-zero it carries the *wrong* sign. The single interval that
excludes zero (forward |return| at 4h) is wrong-signed and is one of twelve
tests, so it should be read as noise.

Directional predictive power is absent outright: ICs of −0.004 to −0.018 on
forward returns, every interval covering zero.

### What counts as a good IC

Worth stating plainly, because the number is easy to misjudge. The familiar
benchmarks (0.02–0.03 weak, 0.03–0.05 solid, above 0.05 strong) come from
CROSS-SECTIONAL equity factors, where hundreds of simultaneous bets let a tiny
edge compound. This is a single-asset TIME-SERIES signal, so breadth is far
lower and the same IC is worth much less. And for volatility specifically the
benchmark is not zero — trailing realized volatility alone predicts forward
realized volatility with an IC around 0.5–0.7, so that is the bar GEX has to
clear. It adds nothing measurable on top.

### What this means

**For a trading model: no.** The incremental IC is indistinguishable from zero
at every horizon, directional content is absent, and an edge of that size on
one asset would not survive Deribit fees and slippage. H6 can be run for
completeness but its answer is already visible here.

**For the thesis: this is a result, not a failure.** The original plan states
that "mechanism real, alpha absent" is a legitimate finding. This is precisely
that finding, quantified — and it is coherent with everything else: the
mechanism shows up contemporaneously (H2) and in event response (H3, post-ETF),
but carries no forecasting content. That is what you would expect if dealer
hedging is a TRANSMISSION channel — it shapes how information already in the
market gets expressed in price — rather than an INFORMATION channel that
anticipates anything.

## 7. Next steps, in order

**1. Decide what the paper claims.** The evidence now supports a narrower
thesis than planned: *the flow-based measure is constructible and behaves as
theory predicts in a small, robust way, while the direct hedging channel cannot
be identified from this data.* That is a defensible contribution — a clean
measurement paper with an honest negative on the mechanism — but it is not the
one the plan was written around, and the framing needs to change before the
deck is built.

**2. Try to rescue H3's identification.** The obstacle is that dealer gamma sign
changes on only 18.7% of days, so short- and long-gamma events sit on different
days. Two routes worth trying, in order of promise:
   - Condition on *changes* in gamma within a day rather than its level, using
     the intraday variation that H2 successfully exploits.
   - Restrict to the days around monthly expiries, where gamma swings hardest
     and fastest — the calendar to do this now exists (section 4.8).
   If neither works, H3 is a null and should be written up as one.

**3. Get the ScienceDirect paper read in full.** The abstract confirms it uses
OI-based gamma and measures expiry-window activity, so the flow-reconstruction
angle survives. But it also reports rising perpetual-futures activity around
expiry, which is closer to H3 than the plan assumed. Confirm the exact gap
before claiming novelty.

**4. Build the deck around H2 and the construction work.** The measurement
chain, the two specification faults found and fixed, and the identification
problem in H3 are all genuinely interesting and all defensible. Lead with them.

**5. Optional, if time allows.** The naive-GEX horse race is now possible
*going forward* — the collector has been capturing open interest and gamma
since 31 August 2026. Three days is not a sample, but it is enough to
demonstrate the comparison works end to end.

**6. Still blocked.** Historical open interest via Tardis (manual download),
which is what H1 and H5 need.

## 8. How the original risks played out

| Risk we predicted | What happened |
|---|---|
| No historical open interest — likely | **Happened.** Cost us nothing, because we deliberately built the parts that don't need it first. Good call. |
| No signal in H2 | **Partly.** After correcting two construction faults the sign is as predicted and borderline significant, but it rests on 2023 alone — section 6. |
| The other paper already did H3 | **Still unchecked.** Now the biggest open risk. |
| Scope creep | Avoided. |
| 90 GB futures download | Avoided — 2.2 GB by filtering properly. |

**Two risks we didn't predict:**
- No historical gamma data (section 4.3). Dealt with.
- **Silent specification error.** Nothing in the pipeline failed, no test broke,
  and 44 tests stayed green while the headline result was being driven by a
  variable that trends. Three of the defects found in the audit — the raw-level
  sort, the block-trade exclusion, the clock-shaped expiry columns — were all
  of this kind: defensible-looking choices, no error message, wrong answer.
  Both belong on the register, and the second argues for spending some of the
  remaining time on diagnostics that would have caught it (what is actually in
  each bucket, does this column vary the way it should) rather than on more
  tests.
