# Why the H2 result came out backwards — diagnostic

**Date:** 1 September 2026. All numbers below are reproducible from the parquet
files already in `data/processed/`.

## Verdict

The section-6 conclusion ("the most short-gamma bucket is the calmest — wrong
sign, and not noise") is **an artifact of pooling four years of an untransformed,
trending dollar series into one cross-sectional sort.** It is not evidence
against the theory.

The GEX arithmetic itself is correct. Four separate suspects were tested and
cleared. Three genuine defects were found; one of them (block trades) changes
the measure by a factor of 2.6.

---

## 1. The H2 result is a calendar-time sort  (fatal to section 6)

`h2_volatility.quintile_sort` calls `pd.qcut(df["gex_flow"], 5)` on the **raw
dollar level**, pooled over 2022-2025. GEX drifts hard over that window:

| year | median GEX | mean spot | % of that year landing in Q1 |
|---|---|---|---|
| 2022 | −$0.61M | $28,226 | 0.1% |
| 2023 | −$3.20M | $28,811 | 3.9% |
| 2024 | −$8.08M | $65,894 | 27.1% |
| 2025 | −$13.05M | $101,638 | 48.8% |

So the quintiles are years in disguise: **Q1 is 61% 2025 + 34% 2024; Q4 is 55%
2022.** 50.1% of the variance of `gex_flow` is between-month.

Part of the drift is mechanical: GEX = pos × gamma × S² × 0.01 and gamma ∝ 1/S,
so **GEX scales linearly with spot**, which rose 3.6x. The rest is book growth.

The whole "signal" is the product of two trends:

```
corr(gex_flow, year) = -0.361
corr(realized_vol, year) = -0.135
product                    = +0.049
observed corr(gex_flow, realized_vol) = +0.032
```

### The contrast, before and after detrending

Top-minus-bottom quintile realized vol, same block bootstrap over days.
Prediction is NEGATIVE (long gamma = calmer).

| specification | Q5 − Q1 | 95% CI | verdict |
|---|---|---|---|
| **raw level** (current) | **+0.000443** | [+0.000133, +0.000743] | wrong sign, excludes zero |
| within-month z-score | −0.000207 | [−0.000494, +0.000094] | predicted sign, includes zero |
| within-year z-score | −0.000017 | [−0.000376, +0.000332] | predicted sign, includes zero |
| rolling-60d z (causal) | −0.000113 | [−0.000407, +0.000168] | predicted sign, includes zero |

**All three detrended specifications flip to the predicted sign and all three
include zero.** The honest statement is *"no detectable relation once the time
trend is removed"*, not *"the theory is backwards"*.

Using forward realized vol instead of trailing changes nothing material
(raw-level Q5−Q1 goes +4.43bp → +4.70bp), so the trailing-window choice is not
what is driving this.

### The residual hump tracks |GEX|, not signed GEX

After within-month detrending the middle quintiles are still the wildest — but
those are the quintiles with the *smallest* |z|. Sorting directly on |GEX|
gives a monotone decline (46.8 → 40.7 bp). The hump is a magnitude effect.

---

## 2. Block trades: 1.3% of trades, but 67% of the net dealer signal

`construction.block_trades.baseline: "exclude"`. The log justifies it with
`block_fraction` computed **by trade count**. By the quantity that actually
matters it is not small (2024 option trades):

| measure | block share |
|---|---|
| trade count | 1.28% |
| **volume (BTC)** | **31.07%** |
| **net dealer flow** | **67.2%** (−85,939 of −127,858 BTC) |

Mean block size is 38.9 BTC vs 1.12 BTC for regular trades. Re-running the full
2024 pipeline both ways:

| | blocks excluded | blocks included |
|---|---|---|
| median GEX | −$6.97M | −$18.60M |
| position mass | 2.18bn | 5.41bn |
| dealers short gamma | 84.4% of intervals | **98.0%** |
| |GEX| median ratio | 1.0x | **2.57x** |
| sign flips vs the other | — | **15.1% of intervals** |
| Spearman correlation | 0.741 | |

Quintile assignment agrees only 33–76% of the time. This is the single largest
specification choice in the project and it was made on a misleading statistic.
**Recommend flipping the baseline to `include`** and reporting `exclude` as
robustness.

---

## 3. The expiry-calendar features are degenerate

Deribit lists a BTC option expiry on **1,461 of 1,461 days** in the sample. So
in `build_calendar`:

- `tte_nearest_hours` ∈ [0, 24) always — 93% of its variance is explained by
  hour-of-day dummies alone.
- `is_expiry_day` is exactly the indicator `hour_of_day < 8`. It is 1 on 100%
  of intervals before 08:00 UTC and 0 after.

Both are clock variables, not expiry variables. STATUS §7 step 3 ("split the
analysis by distance to a big expiry") **cannot be run with the current panel**
— it needs a calendar restricted to monthly/quarterly (large-OI) expiries.

---

## 4. Near-expiry gamma dominates, and its scale is an arbitrary constant

`min_tte_days: 0.0041667` (6 minutes) is what stops BS gamma diverging as
T→0, so it directly sets the maximum gamma the model can produce. March 2024
cell-level:

| time to expiry | % of cells | % of gross gamma $ | largest single cell |
|---|---|---|---|
| <1h | 0.35% | 1.39% | **$140,307,800** |
| 1h–6h | 1.78% | 5.45% | $14.1M |
| 6h–1d | 6.32% | 15.30% | $9.2M |
| 1–7d | 21.4% | 33.5% | $4.2M |
| >30d | 46.2% | 15.8% | $0.6M |

Dropping cells with TTE < 1 day: correlation with the full series is only
**0.36**, std falls $4.2M → $2.9M, max |GEX| falls $156M → $8.25M (19x), and
**13.7% of intervals flip sign**. Because there is a daily expiry, this is a
pervasive variance component in *every* interval, not a tail issue.

The reported GEX range of −$486M to +$230M is largely this.

**Recommend:** a sensitivity table over the floor (6min / 1h / 1d) and a
headline series that excludes the final day before expiry.

---

## 5. The flow measure has no external anchor

Four constructions on March 2024 (10-min intervals):

| construction | mean | % intervals negative |
|---|---|---|
| flow (ours: counted sign × net position) | −$4.10M | 87.8% |
| assumed sign × our own \|net position\| | −$4.49M | 86.7% |
| assumed sign × volume-based OI proxy ("naive") | −$59.10M | 95.9% |
| SpotGamma sign convention × OI proxy | +$59.10M | 4.1% |

Correlation of ours with the naive OI construction: **−0.042** (Spearman
−0.068). With the assumed-sign version of our own positions: −0.381.

The *sign* is corroborated — every construction that assumes dealers are short
calls agrees dealers are net short gamma most of the time, as ours does. But
the series is otherwise **uncorrelated with anything standard**. That is the
paper's thesis (net-after-cancellation vs gross), and it is also why nothing
external validates it. It should be stated that way rather than left implicit.

---

## What was checked and found CORRECT

1. **BS gamma** — re-derived independently on 200,000 cells: max absolute
   difference **0.000e+00**. Magnitudes are textbook (ATM 30d gamma 2.8e−05;
   $1,378 per contract per 1% move).
2. **GEX contribution arithmetic** — max difference 5.8e−11.
3. **Sign convention** — corroborated by every alternative construction (§5).
4. **IV staleness (24h)** — suspected of dropping 40% of the book. It does drop
   40% *by instrument count*, but only 5.5% by GEX: going to 30d staleness
   (coverage 62.6% → 101%) moves mean GEX by 5.5%, level correlation 0.9964,
   Spearman 0.9950. **Cleared.** (The instrument-count coverage does correlate
   +0.31 with realized vol, so it belongs in the robustness table, but it is
   not driving the result.)
5. **Pre-2022 truncation** — only 430 of 76,028 in-sample instruments have
   truncated history, though they run to Dec 2022, so 2022 levels are mildly
   understated.

## Recommended actions, in order

1. Make the **detrended sort the headline** for H2 (causal rolling-60d z-score;
   within-month as robustness). Report the raw-level sort explicitly as the
   confounded specification. **This rewrites STATUS §6.**
2. Flip `construction.block_trades.baseline` to `include`. Change
   `apply_block_trade_policy` to log volume share and net-flow share, not just
   count.
3. Rebuild `build_calendar` on monthly/quarterly expiries only. Nothing that
   uses `is_expiry_day` or `tte_nearest_hours` today is measuring what it says.
4. Add a `min_tte_days` sensitivity and an exclude-final-day GEX variant.
5. Normalise GEX (÷ spot at minimum) so levels are comparable across years.
