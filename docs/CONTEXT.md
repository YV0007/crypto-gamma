# CONTEXT.md

Project context for Claude Code. Read fully before writing any code.

---

## 1. What this project is

A quantitative research paper on **flow-based dealer gamma exposure in crypto options**, using free public Deribit data. Deliverable: a slide deck plus a public GitHub repo, submitted as a work sample for an application to the WUTIS Algorithmic Trading division (WU Vienna student investment society).

Standard to hit: a research note a professional quant shop would take seriously. Understated claims, visible error bars, transparent methodology, reproducible code.

**Author constraints:** solo, ~4 weeks, hard data budget of EUR 100 (currently spending EUR 0).

---

## 2. The mechanism (domain background)

When a customer buys an option, a market maker sells it and hedges the delta with futures. As spot moves, delta changes — **that is gamma** — forcing continuous re-hedging. Direction of that forced flow depends on dealer inventory:

| Dealer inventory | On rally | On selloff | Effect |
|---|---|---|---|
| **Long gamma** | Sells futures | Buys futures | **Dampens** volatility (pinning, mean reversion) |
| **Short gamma** | Buys futures | Sells futures | **Amplifies** volatility (trending) |

Dealer gamma predicts the *character* of price action, not direction.

### The measurement problem

Exchanges publish **open interest** (how many contracts exist) but never **who is long vs short**. That sign is unobservable, and the sign is the whole hypothesis.

### How the standard indicator handles it

```
GEX_naive = Σ(call OI × Γ × S² × 0.01) − Σ(put OI × Γ × S² × 0.01)
```

The `−` on puts encodes a hardcoded assumption: *customers buy calls, sell puts; dealers take the other side.* Nothing is measured.

**Two errors:**

1. **Sign error.** In crypto, participants buy calls to speculate on upside rather than selling for yield; miners and treasuries write covered calls persistently. The heuristic signs this backwards. A wrong sign *reverses* the prediction, not just weakens it.
2. **Netting error.** OI at a strike is built from both buying and selling flow. If 50% came from takers buying (dealers short) and 50% from takers selling (dealers long), net dealer exposure is ~zero — but the heuristic still reports large exposure. So it's wrong on magnitude even when the sign is right.

### Our approach

Deribit flags the **taker** (aggressor) on every trade. Market makers sit in the book quoting; customers cross the spread.

```
Taker bought  ->  customer bought  ->  dealer sold   ->  position -1
Taker sold    ->  customer sold    ->  dealer bought ->  position +1
```

Accumulate per instrument from listing. **We count the sign instead of guessing it.**

```
reconstructed dealer position   <- the ONLY thing we build
      × published gamma          <- Deribit provides directly
      = signed gamma exposure    <- our variable
```

---

## 3. Why crypto, why now

- Spot BTC ETFs launched January 2024; institutional participation rose sharply
- Deribit handles the large majority of global crypto options volume; acquired by Coinbase
- Options OI has grown materially relative to spot volume
- **Uniquely observable:** options positioning, taker direction, AND the perpetual futures venue where dealers hedge are all public and on the same exchange

That last point is the methodological key. Equity studies infer hedging through three steps with no direct evidence. On Deribit the hedging flow itself is visible.

---

## 4. Prior work and our gap

| Source | Did | Did not |
|---|---|---|
| **Glassnode (Dec 2025)** — *Taker-Flow-Based Gamma Exposure*, research.glassnode.com/gamma-exposure | Built and shipped flow-based GEX for BTC/ETH/SOL/XRP/PAXG on Deribit, 10-min resolution | **Methodology gated** ("reach out to your account manager"). **Zero statistical testing** — no backtest, no CIs, no validation. Own disclaimer: "a limited case study with significant constraints" |
| **ScienceDirect (2026)** — *Bitcoin option expiration, gamma exposure, and intraday price reversals* (S1544612326008688) | Expiry return reversals, strongest when cumulative GEX negative; ~USD 50m/yr wealth transfer | Uses **OI-based** gamma, not flow-reconstructed. Measures *activity* around expiry, not **signed** hedging flow |

### Seven contributions

1. Transparent construction — publish every choice Glassnode hides; open repo
2. Reconciliation of reconstructed inventory against published OI
3. Statistical validation with proper standard errors
4. Head-to-head horse race vs naive OI-based GEX
5. **Direct test of the hedging channel in signed perpetual flow** (core)
6. Sensitivity analysis on the maker=dealer assumption
7. Tradability net of costs, deflated Sharpe

Framing: *replication + transparency + validation + one new test.* This matches the WUTIS archive — none of their decks claim novelty, all claim competent execution plus a small extension.

---

## 5. Hypotheses

### H2 — Gamma conditions volatility and autocorrelation

High GEX -> lower realized vol, negative intraday autocorrelation. Low/negative GEX -> higher vol, positive autocorrelation.

Test: GEX quintile sort (realized vol, AC(1), AC(5) per quintile, expect monotonicity), plus regression `RV[t+1] = α + β·GEX[t] + γ·controls + ε`, expecting β<0. Controls: DVOL, spot volume, funding rate, hour-of-day.

**Needs no OI.**

### H3 — Hedging channel directly observable (CORE)

Conditional on a large spot move, gamma sign predicts subsequent signed perpetual taker flow.

| Gamma | Spot rallies | Expected perp flow |
|---|---|---|
| Positive | up | Net **selling** |
| Negative | up | Net **buying** |

Test: identify |5-min return| > 1σ events; split by GEX sign; measure net signed perp taker volume over t+1…t+30min; bootstrap CIs.

This is the test nobody has run, and it validates the entire measurement chain — if positioning were noise, conditioning on its sign would produce nothing.

**Needs no OI.**

### H4 — Effect strengthened with institutionalisation

Rolling 90d estimate of H2's β; regress on options OI/spot volume ratio and cumulative ETF flows; Chow test at 2024-01-11.

**Needs no OI** (ETF flows and volume suffice; OI ratio is optional).

### H1 — Inventory reconciliation

Reconstructed positions should sum to published OI. **Needs OI. Not a gate** — it only proves trade capture, not sign accuracy. H3 is the real validation.

### H5 — Expiry pinning

Friday 08:00 UTC expiries; distance from spot to max-OI strike over final 24h; placebo on non-expiry Fridays. **Needs per-strike OI.**

### H6 — Tradability

Mean-revert when GEX high, momentum when low. Deribit taker fees + slippage from the first run. Benchmarks: always-MR, always-momentum, buy-and-hold. **Permitted to fail** — "mechanism real, alpha absent" is a legitimate finding.

**Needs no OI.**

---

## 6. Data — all free, EUR 0

| Data | Endpoint | Purpose |
|---|---|---|
| Instrument universe | `public/get_instruments` (`expired=true` AND `false`, merged) | Enumerate all historical strikes |
| Options trades, taker-flagged | `history.deribit.com/api/v2/public/get_last_trades_by_instrument_and_time` | Dealer inventory |
| Perpetual trades | Same endpoint, `BTC-PERPETUAL` | H3 hedging flow |
| OI, mark IV, greeks | `public/get_book_summary_by_instrument`, `public/ticker` | Gamma; current OI |
| Spot/perp OHLCV | `public/get_tradingview_chart_data` | Returns, realized vol |
| DVOL | `public/get_volatility_index_data` | Control |
| BTC ETF flows | Farside Investors (free web) | H4 |

### Key facts

- **Retention is unlimited.** Records beyond 24h reachable via `historical=true` or `history.deribit.com`. All trades since platform launch are public.
- **API key:** Deribit-generated, **all scopes `none`**. Public endpoints need no scope; key is only for rate-limit headroom (20/s standard, 100/s burst). Credentials in `.env`, `.env` in `.gitignore` **before first commit** — repo goes public.
- **Greeks: do not recompute.** Deribit publishes gamma/delta/vega/theta per instrument. Pull them.

### Existing tooling

`github.com/RiveChen/deribit-historical-data` — async full-history fetcher.

- BTC options: ~1–2 hours, ~10 GB
- BTC futures: ~3–4 hours, **~90 GB -> fork to filter `BTC-PERPETUAL` only**
- Resumable (SQLite checkpoints), rate-limit aware, JSONL->Parquet converter
- Read `docs/api-reference.md` first — documents `trade_seq` chunking quirks

### Sample period

**2022-01-01 -> 2025-12-31.** Two years either side of the ETF launch.

---

## 7. The open interest problem — RESOLVED, and deferred

**Deribit publishes no historical OI.** `get_book_summary_by_instrument` is a current snapshot only. Verified.

### Sources investigated and ruled out

| Source | Verdict |
|---|---|
| Coin Metrics community | 24h only; historical is Pro |
| CryptoDataDownload | Options files are OHLCV — no OI field; also paid |
| Velo | OI coverage is futures/perps, not per-strike options |
| Amberdata / Kaiko / CoinAPI | Have it, all well over EUR 100 |
| Kaggle | Only a snapshot, not a time series |
| Laevitas | Historical OI exists but expiry-aggregated, not per strike |

### Chosen approach

1. **Tardis.dev free monthly samples** (primary). Deribit data from 2019-03-30; **CSVs for the first day of each month downloadable with no API key**. The `ticker` channel (since 2019-10-01) includes price, volume, **open interest**, funding, and Greeks for options. -> ~48 strike-level snapshots across the sample.
2. **Gross accumulated volume proxy** from own trade data, for a full-frequency naive benchmark. Validate against Tardis snapshots (regress proxy on true OI, report R²), disclose in methods.
3. **Own daily collector** from today forward — the only irreplaceable task; every day not collected is lost.
4. **Optional:** BackQuant API ($49/mo, 7-day free trial) has historical GEX time series and options OI. Caveat: multi-venue aggregated (Deribit + Bybit + OKX + Binance + others) and undisclosed methodology, so it contaminates a clean horse race. Useful as third-party cross-validation only.

### Sequencing decision

**Build everything OI-free first.** H2, H3, H4, H6 and the entire core pipeline need no OI. The benchmark (naive GEX), H5, and H1 get bolted on in weeks 3–4 once OI is secured.

---

## 8. Known limitations — must appear in the deck, not be hidden

1. **Taker != customer.** Retail and funds also post limit orders. This is a proxy. Glassnode states the same assumption flatly with no sensitivity analysis; we sweep it.
2. **Aggressor != open/close.** A buy-aggressor may open a long or close a short. Inventory is an approximation. Deribit's public feed has no open/close flag — that's exactly what Cboe sells for ~$600/month in equities.
3. **OI reconciliation validates data capture, not sign accuracy.** A contract has the same OI regardless of who is long. **H3 is the real validation of the sign.**
4. **Deribit != global market.** OKX, Bybit, CME also list crypto options.
5. **Not all market makers delta-hedge.** Some run directional books.
6. **Endogeneity.** Volatility influences what customers trade, which influences gamma. H3 is the primary defence — it tests the mechanism, not the correlation.

---

## 9. Statistical standards — non-negotiable

| Requirement | Implementation |
|---|---|
| Autocorrelation | Newey-West SEs everywhere (overlapping expiries) |
| Multiple testing | **Deflated Sharpe Ratio** (Bailey & López de Prado); log every specification tried |
| Out-of-sample | Declared in methods before results. Train 2022–23, test 2024–25 |
| Costs | Deribit fee schedule + slippage, from the first backtest |
| Uncertainty | CIs on every estimate; no naked point estimates |
| Reproducibility | Fixed seeds, `config.yaml`, every figure regenerable by script |
| Transparency | Public GitHub repo linked on the title slide |

Target register: match the WUTIS QNN deck's "WE DO NOT CLAIM" slide. If the effect is real but untradeable, say so — their 2024 short straddle deck reported an underperformance and still made the archive.

---

## 10. Engineering conventions

- **Notebooks explore, `src/` produces.** Every figure in the deck must be regenerable by running a script.
- Raw data in `data/raw/` is **never edited**. Cleaned -> `data/interim/`, analysis-ready -> `data/processed/`.
- All parameters in `config.yaml`, not hardcoded.
- Parquet for anything over ~10k rows.
- Type hints and docstrings on public functions.
- Every construction choice (block trade handling, decay, expiry, filters) is a config parameter with a documented baseline and testable alternatives — **this transparency is contribution #1**.

### Documented construction choices

| Choice | Baseline | Robustness alternatives |
|---|---|---|
| Block trades | Exclude (`block_trade_id` flagged) | Include; 0.5 weight |
| Position decay | None (pure accumulation) | Exponential, 30d half-life |
| Expiry | Hard zero at expiry timestamp | — |
| Deep OTM | Include all | Exclude \|delta\| < 0.01 |
| Interval | 10-min (matches Glassnode) | 1h, daily |

---

## 11. Core formulas

**Dealer inventory:**

```python
customer_net_flow[i,t] = taker_buy_vol - taker_sell_vol
dealer_flow[i,t]       = -customer_net_flow[i,t]
dealer_position[i,t]   = dealer_position[i,t-1] + dealer_flow[i,t]
# initialise 0 at listing, zero at expiry
```

**Flow-based GEX:**

```python
GEX_flow[t] = Σ_i dealer_position[i,t] × Γ[i,t] × S[t]² × 0.01 × contract_size
```

**Naive GEX (benchmark, later):**

```python
GEX_naive[t] = Σ_i sign_i × OI[i,t] × Γ[i,t] × S[t]² × 0.01 × contract_size
# sign_i = -1 calls, +1 puts
```

Units: notional USD per 1% spot move.

---

## 12. Extensions (appendix — cut first if time runs short)

- **Quanto adjustment.** Deribit BTC options are inverse: quoted/margined in BTC, payoff referencing USD. The dealer's USD-hedging delta differs from naive Black-Scholes because the numéraire moves with the underlying. Original derivatives mathematics; neither prior source addresses it.
- **Vanna and charm decomposition.** Vanna (dDelta/dSigma) drives flow when IV moves; charm (dDelta/dt) dominates into short-dated expiry. Horse-race which Greek drives flow, and when.

---

## 13. Timeline

**Week 1** — start daily OI collector (day 1); get the ScienceDirect paper via WU library to confirm H3 is open; clone fetcher; run BTC options backfill; build inventory for one month; first sanity checks.

**Week 2** — full-period inventory and GEX_flow; pull BTC-PERPETUAL (filtered); sensitivity sweep; **stylized-fact chart: GEX quintiles vs realized vol**. If nothing visible in the raw sort, stop and reassess.

**Week 3** — H2 full spec; **H3 event study (core)**; H4; secure OI and add naive benchmark + horse race; H5.

**Week 4** — H6 + costs + deflated Sharpe; robustness battery; build deck; clean repo; public README.

---

## 14. Deck outline (15 slides + appendix)

1. Title (+ GitHub link)
2. Motivation & prior work — mechanism diagram, Glassnode + ScienceDirect, explicit gap
3. Why crypto, why now — ETF flows, OI growth, unique observability
4. Data & construction — sources, sample, filters, obs surviving each, **all choices disclosed**
5. Naive GEX and why it fails — sign error + netting error, worked example
6. H1 reconciliation — error distribution
7. Sensitivity — maker=dealer misclassification sweep
8. Stylized facts — raw quintile sort, no model
9. H2 — vol & autocorrelation, table with CIs
10. **H3 — direct hedging evidence** (core slide)
11. Horse race — flow vs naive
12. H4 — growth over time, rolling β, Chow test
13. H5 — pinning vs placebo
14. H6 — economic significance, net of costs, deflated Sharpe
15. **What we do not claim** — limitations, what would falsify

A1+ Appendix — quanto derivation, vanna/charm, robustness tables

---

## 15. References

- Glassnode, Lamblin & Abril (2025). *Introducing: Taker-Flow-Based Gamma Exposure.* research.glassnode.com/gamma-exposure
- (2026). *Bitcoin option expiration, gamma exposure, and intraday price reversals.* ScienceDirect S1544612326008688
- Ni, Pearson & Poteshman. *Stock price clustering on option expiration dates.*
- Barbon & Buraschi. Dealer hedging and market dynamics.
- Baltussen, Terstegge & Da. Gamma exposure and volatility.
- Bailey & López de Prado. *The Deflated Sharpe Ratio.*
- López de Prado. *Advances in Financial Machine Learning.*
- Deribit API: docs.deribit.com
- Fetcher: github.com/RiveChen/deribit-historical-data
- Tardis Deribit coverage: docs.tardis.dev/historical-data-details/deribit
