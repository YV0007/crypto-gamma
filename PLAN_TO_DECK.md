# Everything to do before the deck

> **EXECUTED 4 September 2026.** Phases 1-4 complete. Phase 2 resolved the H3
> reversal (it is 2022, a small-gamma measurement regime, not a venue effect).
> All 19 figures and 18 tables current; 177 tests pass. Phase 5 (content
> decisions) is the only part left, and it is writing, not code.

**Written 4 September 2026.** Scope: only work that does not depend on data we
cannot get. H1, H5 and the naive horse race are excluded throughout — they need
historical open interest, which does not exist publicly. ETF-options GEX is
excluded for the same reason.

Order matters. Phase 1 fixes things that are currently **wrong**. Phase 2 is
the one open scientific question. Phase 3 produces what the deck needs. Phase 4
makes sure the numbers agree with each other.

---

## Phase 1 — Fix what is actively wrong  (half a day)

Five outputs on disk pre-date the block-trade fix that landed 1 Sep 07:49, so
they are built on a GEX series that discarded 97% of the net dealer signal.
Anyone opening them today gets the wrong picture.

| # | task | file | why |
|---|---|---|---|
| 1.1 | regenerate full-period GEX chart | `gex_flow_full_period.png` | built 31 Aug, blocks excluded |
| 1.2 | regenerate March 2024 GEX chart | `gex_flow_2024-03.png` | same |
| 1.3 | regenerate inventory lifecycle | `inventory_lifecycle_2024-03.png` | same |
| 1.4 | regenerate H3 hedging chart | `h3_hedging_flow.png` | pre-dates the corrected H3 run |
| 1.5 | delete superseded H2 figures | `h2_quintile_*.png` (unsuffixed) | replaced by `_rolling_z` variants; risk of grabbing the wrong one |

**Done when:** no figure in `outputs/figures/` is older than the 1 Sep rebuild.

---

## Phase 2 — Resolve the H3 pre-ETF reversal  (1–2 days)

The only open scientific question. H3 is significantly **positive** after
January 2024 and significantly **negative** before, under every identification
strategy. A mechanical hedging obligation cannot reverse, so one of two things
is true and the deck cannot say which:

  (a) dealers did not hedge on Deribit before 2024 — inventories were 3x
      smaller and hedging plausibly went to CME, spot, or was internalised; or
  (b) the pre-2024 reconstruction is unreliable — thin market, and 430
      instruments alive at the sample start have truncated histories.

This is worth doing because it decides whether slide 10 says *"the mechanism is
observable post-2024"* (strong) or *"our result is unstable"* (weak).

| # | task | tests which |
|---|---|---|
| 2.1 | pre-2024 data-quality audit: gamma coverage, instrument counts, share of GEX from truncated instruments, IV staleness by year | (b) |
| 2.2 | drop the 430 truncated instruments, re-run the pre-ETF DiD | (b) |
| 2.3 | restrict pre-ETF to high-\|gamma\| events only — if the reversal is noise from tiny positions it should weaken | (b) |
| 2.4 | expiry-conditioned H3, using the rebuilt major-expiry calendar. The one crypto paper that finds real effects localises them to expiry windows | (a) |
| 2.5 | Deribit perp volume share by year — if Deribit perps were a small hedging venue pre-2024, that supports (a) | (a) |

**Done when:** the deck can state one of the two explanations, or state plainly
that both were tested and neither resolves it.

**If time is short:** skip to Phase 3 and present H3 as post-2024 only, with the
reversal in the limitations slide. Costs one slide of strength, not correctness.

---

## Phase 3 — Build the six missing deck outputs  (1 day)

Every one is scripted from data already on disk.

| # | output | slide | source |
|---|---|---|---|
| 3.1 | **sample funnel table** — observations surviving each filter, from raw trades to final panel | 4 · Data & construction | pipeline logs + `apply_block_trade_policy` stats |
| 3.2 | **gamma validation scatter** — our gamma vs Deribit's published, r = 0.996 | 5 · Why the measure is trustworthy | `gamma_aggregate_validation.json` |
| 3.3 | **H3 era-split chart** — pre vs post ETF across every identification strategy | 10 · H3 core | `h3_identification.json` |
| 3.4 | **H4 chart** — beta either side of the break, with intervals | 12 · H4 | `h4_results.json` |
| 3.5 | **IC chart** — raw vs incremental, by horizon | 14 · Prediction | `signal_ic.csv` |
| 3.6 | **multiple-testing chart** — 7 tests, q-values, 2 survive | 15 · What we don't claim | `multiple_testing.csv` |

3.2 is the most valuable single addition: it is the slide that answers *"why
should I believe your numbers"* and it currently exists only as JSON.

---

## Phase 4 — Consistency and reproducibility  (half a day)

Results were regenerated several times as faults were found. Numbers quoted in
`STATUS.md` and `DIAGNOSTIC_H2.md` were written at different points and may no
longer match what is in `outputs/`.

| # | task |
|---|---|
| 4.1 | audit every number in `STATUS.md` against current `outputs/tables/` — fix or flag each mismatch |
| 4.2 | same for `DIAGNOSTIC_H2.md` |
| 4.3 | `scripts/reproduce_all.sh` — one command, raw data to every figure and table, in order |
| 4.4 | full test suite green (177 currently) |
| 4.5 | **key-numbers sheet** — one file holding every figure that appears on a slide, so the deck is never typed from memory |

4.5 matters more than it looks. Several numbers in this project have changed
two or three times; a slide typed from an old message would be wrong.

---

## Phase 5 — Deck content decisions  (not code)

Settle before writing slides:

- **The claim.** Current evidence supports: *a validated flow-based measure of
  dealer gamma; a small but robust volatility relationship; direct hedging
  visible post-2024; no forecasting power and no tradable edge.* That is a
  measurement paper with honest nulls, not the alpha story the plan opened with.
- **Cut slides 6, 11, 13** (H1, horse race, H5) — blocked on data.
- **Rewrite slide 5.** It was "naive GEX and why it fails", which needed the
  horse race. It becomes "why our measure is trustworthy", carrying the 99.2%
  aggregate reconciliation instead. Stronger, and it is evidence we have.
- **Slide 15 must name three limitations:** no historical open interest; roughly
  half of bitcoin dealer gamma now sits in IBIT options outside the measure;
  the H3 pre-2024 reversal.

---

## Critical path

```
Phase 1  (wrong -> right)        half day
Phase 3  (missing -> built)      1 day        <- deck is buildable after this
Phase 4  (consistency)           half day
                                 ----------
                                 2 days to a deck that is correct

Phase 2  (H3 reversal)           1-2 days     <- optional, buys a stronger slide 10
```

Phase 2 is the only judgement call. Do it if there is a week; skip it and
document the reversal if there are days.
