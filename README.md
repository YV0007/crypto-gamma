# Flow-Based Dealer Gamma in Crypto Options

Reconstructing dealer gamma exposure from Deribit **taker flow** rather than
from open interest and a sign assumption, and testing whether it predicts
realized volatility, intraday autocorrelation, and — uniquely observable on
Deribit — dealer hedging flow in perpetual futures.

Full project context, hypotheses and limitations: [`CONTEXT.md`](CONTEXT.md).

---

## The idea in three lines

The standard GEX indicator assumes customers buy calls and sell puts, and signs
dealer inventory accordingly. Deribit flags the **taker** on every trade, so the
sign can be counted instead of assumed:

```
taker bought -> customer bought -> dealer sold   -> position -1
taker sold   -> customer sold   -> dealer bought -> position +1
```

---

## Status

**Stage 1 (data acquisition) is implemented and running.** Stages 2+ (`src/build`,
`src/validate`, `src/analysis`) are not yet written.

---

## Quickstart

```bash
python3 -m src.fetch.instruments          # instrument universe -> data/interim/
./scripts/run_backfill.sh                 # options -> perp -> market data
python3 scripts/status.py                 # progress, safe to run any time
python3 scripts/to_parquet.py             # raw JSONL -> partitioned parquet
python3 -m pytest tests/ -q
```

No credentials are required — every endpoint used is public. `.env` is optional
and buys only rate-limit headroom; see `.env.example`.

---

## Verified API behaviour

Reconnaissance against the live API on 2026-08-31 (`scripts/probe_api.py`,
results in `outputs/tables/api_recon.json`). Four findings materially shaped the
implementation:

| Finding | Consequence |
|---|---|
| `get_instruments(expired=true)` returns **56** instruments on `www.deribit.com` but **120,929** (back to 2016-07-15) on `history.deribit.com` | Instruments must be enumerated against the **history** host. Using `www` silently truncates the sample to the current expiry batch. |
| `get_last_trades_by_currency_and_time` returns every option trade in a window with `instrument_name` embedded | ~5k requests for the full sample instead of ~120k. Proven lossless against per-instrument `trade_seq` paging by `scripts/validate_endpoint_equivalence.py`. |
| `get_volatility_index_data` returns **zero points, without erroring**, for `resolution` `1` and `60` | DVOL resolution is pinned to `3600` in config and asserted in tests. |
| Options report `contract_size` **1.0**; `BTC-PERPETUAL` reports **10.0** | The per-instrument value is carried in `instruments.parquet` and must be joined from there. The config key is named `contract_size_fallback` so nothing reaches for it by habit. |

### Deviation from the RiveChen fetcher

[`RiveChen/deribit-historical-data`](https://github.com/RiveChen/deribit-historical-data)
pages per-instrument on `trade_seq`. This project pages **time windows at
currency level** instead, because it removes the need to enumerate 120,929
instruments. Their dedup rule still applies and is still needed — Deribit
returns overlapping rows at page boundaries (49 duplicates in 555,365 rows for
2024-03), removed on `(instrument_name, trade_seq)` at parquet export.

Equivalence was verified rather than assumed: 12 instruments spanning 2 to 5,615
trades, exact `trade_id` set match, zero missing and zero extra.

---

## Layout

```
config.yaml            every parameter; nothing is hardcoded
CONTEXT.md             thesis, hypotheses, prior work, limitations

src/fetch/             Stage 1 -- implemented
  client.py              async httpx, token-bucket rate limit, backoff+jitter
  runner.py              day-partitioned resumable backfill, atomic writes
  instruments.py         universe + per-instrument contract_size
  option_trades.py       taker-flagged option trades
  perp_trades.py         BTC-PERPETUAL only (never all futures, ~90 GB)
  market_data.py         OHLCV 1m + DVOL
  book_summary.py        OI/greeks SNAPSHOT ONLY -- no historical OI exists
  checkpoint.py          SQLite progress, per-day granularity

src/build/             Stage 2 -- not yet implemented
src/validate/          H1 reconciliation, maker=dealer sensitivity
src/analysis/          H2..H6, horse race
src/utils/             Newey-West, deflated Sharpe, plotting

scripts/
  run_backfill.sh                    sequential driver, resumable
  status.py                          progress summary
  to_parquet.py                      JSONL -> partitioned parquet, deduped
  validate_month.py                  Stage-1 acceptance checks
  validate_endpoint_equivalence.py   proves the endpoint deviation is lossless
  probe_api.py                       API reconnaissance
```

`data/raw/` is append-only and never edited. Cleaned data goes to
`data/interim/`, analysis-ready panels to `data/processed/`.

---

## Resumability

One UTC day is the atomic unit of work. A day is fetched in full, written to a
temp file, fsynced, atomically renamed, and only then marked `done` in
`data/raw/.checkpoints.db`. An interrupted run can never record a partial day.

`SIGINT`/`SIGTERM` stops new days being handed out and lets in-flight days
finish, so files and checkpoints always agree. Rerunning any stage resumes;
failed days are retried automatically on the next run.

---

## The open-interest constraint

Deribit publishes **no historical open interest** — `get_book_summary_by_*` and
`ticker` are current-state only, on both hosts. This is why H2, H3, H4 and H6 are
built first: none of them need OI. The naive-GEX benchmark, H1 reconciliation and
H5 pinning depend on OI and come later, from Tardis.dev monthly samples plus a
forward-looking daily collector (`src/fetch/book_summary.py`). See
`CONTEXT.md` §7.
