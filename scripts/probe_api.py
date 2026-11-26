"""Read-only reconnaissance of the Deribit public API.

Answers the Stage-1 recon questions before any fetcher is written:
  (a) expired-instrument coverage -- does it reach the sample start?
  (b) trade history on an expired option -- are `direction` and
      `block_trade_id` present?
  (c) greeks and open_interest on a live option
  (d) actual `contract_size` values (config.yaml assumes 1.0)
  (e) volume-based size/runtime estimate for the full sample

Writes nothing except a JSON summary under outputs/tables/.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

BASE = "https://www.deribit.com/api/v2"
HISTORY = "https://history.deribit.com/api/v2"
SAMPLE_START = datetime(2022, 1, 1, tzinfo=timezone.utc)


def ts(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def call(client: httpx.Client, host: str, method: str, **params):
    r = client.get(f"{host}/public/{method}", params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()
    if "error" in payload:
        raise RuntimeError(f"{method}: {payload['error']}")
    return payload["result"]


def main() -> int:
    out: dict = {}
    with httpx.Client(headers={"User-Agent": "crypto-gamma-recon/0.1"}) as c:
        # ---------------- (a) instrument universe -------------------------
        live = call(c, BASE, "get_instruments", currency="BTC", kind="option", expired="false")
        expired = call(c, BASE, "get_instruments", currency="BTC", kind="option", expired="true")
        allinst = live + expired
        print(f"(a) live options            : {len(live)}")
        print(f"    expired options returned: {len(expired)}")
        print(f"    total                   : {len(allinst)}")

        created = sorted(i["creation_timestamp"] for i in allinst)
        expiries = sorted(i["expiration_timestamp"] for i in allinst)
        earliest_creation = ts(created[0])
        earliest_expiry = ts(expiries[0])
        print(f"    earliest creation_timestamp : {earliest_creation.isoformat()}")
        print(f"    earliest expiration_timestamp: {earliest_expiry.isoformat()}")
        print(f"    latest expiration_timestamp  : {ts(expiries[-1]).isoformat()}")
        reaches = earliest_creation <= SAMPLE_START
        print(f"    COVERAGE REACHES 2022-01-01 : {reaches}")
        out["a_instruments"] = {
            "live": len(live),
            "expired_returned": len(expired),
            "total": len(allinst),
            "earliest_creation": earliest_creation.isoformat(),
            "earliest_expiry": earliest_expiry.isoformat(),
            "latest_expiry": ts(expiries[-1]).isoformat(),
            "reaches_sample_start": reaches,
        }

        # how many expired instruments fall inside the sample window?
        in_sample = [
            i for i in allinst
            if SAMPLE_START <= ts(i["expiration_timestamp"]) <= datetime(2026, 1, 1, tzinfo=timezone.utc)
        ]
        print(f"    instruments expiring within 2022-01-01..2025-12-31: {len(in_sample)}")
        out["a_instruments"]["in_sample_window"] = len(in_sample)

        # ---------------- (d) contract_size -------------------------------
        sizes = Counter(i.get("contract_size") for i in allinst)
        print(f"(d) contract_size distribution: {dict(sizes)}")
        by_kind = Counter((i.get("kind"), i.get("contract_size")) for i in allinst)
        print(f"    by kind: {dict(by_kind)}")
        perp = call(c, BASE, "get_instruments", currency="BTC", kind="future", expired="false")
        perp_cs = {p["instrument_name"]: p.get("contract_size") for p in perp
                   if p["instrument_name"] == "BTC-PERPETUAL"}
        print(f"    BTC-PERPETUAL contract_size: {perp_cs}")
        out["d_contract_size"] = {
            "options": {str(k): v for k, v in sizes.items()},
            "perpetual": perp_cs,
        }

        # ---------------- (b) trade history on an expired option ----------
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        six_mo_ms = now_ms - 182 * 86_400_000
        cands = [i for i in expired if abs(i["expiration_timestamp"] - six_mo_ms) < 30 * 86_400_000]
        cands.sort(key=lambda i: abs(i["expiration_timestamp"] - six_mo_ms))
        probe_inst = None
        trades: list = []
        for cand in cands[:40]:
            name = cand["instrument_name"]
            res = call(c, HISTORY, "get_last_trades_by_instrument",
                       instrument_name=name, count=100, start_seq=1, end_seq=10_000)
            t = res.get("trades", [])
            if t:
                probe_inst, trades = cand, t
                break
        if probe_inst is None:
            print("(b) NO expired option with trades found among candidates")
            out["b_trades"] = {"ok": False}
        else:
            name = probe_inst["instrument_name"]
            print(f"(b) probe instrument: {name} "
                  f"(expired {ts(probe_inst['expiration_timestamp']).date()})")
            print(f"    trades returned in first chunk: {len(trades)}")
            print(f"    fields present: {sorted(trades[0].keys())}")
            has_dir = all("direction" in t for t in trades)
            dirs = Counter(t.get("direction") for t in trades)
            n_block = sum(1 for t in trades if t.get("block_trade_id"))
            seqs = [t["trade_seq"] for t in trades]
            print(f"    `direction` on every row: {has_dir}  distribution: {dict(dirs)}")
            print(f"    `block_trade_id` present on: {n_block}/{len(trades)} rows "
                  f"(field in schema: {'block_trade_id' in trades[0] or n_block > 0})")
            print(f"    trade_seq range in chunk: {min(seqs)}..{max(seqs)}  has_more={res.get('has_more')}")
            print(f"    sample row: {json.dumps(trades[0], indent=2)[:900]}")
            out["b_trades"] = {
                "ok": True,
                "instrument": name,
                "n_first_chunk": len(trades),
                "fields": sorted(trades[0].keys()),
                "direction_on_every_row": has_dir,
                "direction_dist": {str(k): v for k, v in dirs.items()},
                "rows_with_block_trade_id": n_block,
                "has_more": res.get("has_more"),
            }

        # ---------------- (c) ticker greeks on a live option --------------
        live_sorted = sorted(live, key=lambda i: i["expiration_timestamp"])
        tick = None
        for cand in live_sorted[:50]:
            t = call(c, BASE, "ticker", instrument_name=cand["instrument_name"])
            if t.get("greeks"):
                tick = (cand["instrument_name"], t)
                break
        if tick:
            name, t = tick
            g = t.get("greeks", {})
            print(f"(c) ticker probe: {name}")
            print(f"    greeks keys   : {sorted(g.keys())}")
            print(f"    gamma={g.get('gamma')} delta={g.get('delta')} "
                  f"vega={g.get('vega')} theta={g.get('theta')}")
            print(f"    open_interest={t.get('open_interest')} mark_iv={t.get('mark_iv')} "
                  f"underlying_price={t.get('underlying_price')}")
            out["c_ticker"] = {
                "instrument": name,
                "greeks_keys": sorted(g.keys()),
                "has_gamma": "gamma" in g, "has_delta": "delta" in g,
                "has_vega": "vega" in g, "has_theta": "theta" in g,
                "has_open_interest": t.get("open_interest") is not None,
                "has_mark_iv": t.get("mark_iv") is not None,
            }
        else:
            print("(c) no greeks found on probed live options")
            out["c_ticker"] = {"ok": False}

        # ---------------- (e) size / runtime estimate ---------------------
        # last_seq per instrument == total trades ever on it. Sample a slice
        # of in-sample instruments and extrapolate.
        sample = in_sample[:: max(1, len(in_sample) // 60)][:60]
        counts = []
        for inst in sample:
            res = call(c, HISTORY, "get_last_trades_by_instrument",
                       instrument_name=inst["instrument_name"], count=1)
            tr = res.get("trades", [])
            counts.append(tr[0]["trade_seq"] if tr else 0)
        if counts:
            mean_trades = sum(counts) / len(counts)
            est_total = mean_trades * len(in_sample)
            BYTES_PER_TRADE = 420          # measured on the JSONL sample below
            est_gb = est_total * BYTES_PER_TRADE / 1e9
            est_reqs = est_total / 10_000 + len(in_sample)
            est_hours = est_reqs / 15 / 3600
            print(f"(e) sampled {len(counts)} in-sample instruments")
            print(f"    mean trades/instrument : {mean_trades:,.0f}")
            print(f"    median                 : {sorted(counts)[len(counts)//2]:,}")
            print(f"    ESTIMATED TOTAL TRADES : {est_total:,.0f}")
            print(f"    ESTIMATED RAW JSONL    : {est_gb:,.1f} GB")
            print(f"    ESTIMATED REQUESTS     : {est_reqs:,.0f}")
            print(f"    ESTIMATED RUNTIME @15/s: {est_hours:,.1f} h")
            out["e_estimate"] = {
                "sampled": len(counts), "mean_trades": mean_trades,
                "est_total_trades": est_total, "est_raw_gb": est_gb,
                "est_requests": est_reqs, "est_hours_at_15rps": est_hours,
            }
        if trades:
            b = len(json.dumps(trades[0]).encode())
            print(f"    measured bytes/trade (JSONL): {b}")
            out["e_estimate"]["measured_bytes_per_trade"] = b

    dest = Path("/Users/macbook/Desktop/crypto-gamma/outputs/tables/api_recon.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
