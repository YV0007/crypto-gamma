"""TASK 2 -- is get_order_book's open_interest on an EXPIRED option usable?

Timeboxed check. Terminal OI is ONE value per contract at settlement, not a
time series -- see the report at the bottom for what it does and does not
unblock.
"""
from __future__ import annotations
import sys
from pathlib import Path
import httpx, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

BASE = "https://www.deribit.com/api/v2"

inst = pd.read_parquet(config.path("paths.interim") / "instruments.parquet")
opts = inst[(inst["kind"] == "option") & (inst["expiry_ts"] < pd.Timestamp.now(tz="UTC"))]

# 20 instruments spread across years and expiries
opts = opts.copy(); opts["year"] = opts["expiry_ts"].dt.year
picks = (opts.groupby("year", group_keys=False)
         .apply(lambda g: g.sample(min(5, len(g)), random_state=config.get("seed", 42)))
         .head(20))

rows = []
with httpx.Client(headers={"User-Agent": "Mozilla/5.0 crypto-gamma"}, timeout=40) as c:
    for _, i in picks.iterrows():
        n = i["instrument_name"]
        try:
            r = c.get(f"{BASE}/public/get_order_book",
                      params={"instrument_name": n, "depth": 1}).json()
            if "error" in r:
                rows.append({"instrument": n, "expiry": i["expiry_ts"], "err": r["error"].get("message")})
                continue
            d = r["result"]
            rows.append({
                "instrument": n, "expiry": i["expiry_ts"], "strike": i["strike"],
                "type": i["option_type"], "open_interest": d.get("open_interest"),
                "settlement": d.get("settlement_price"), "mark_iv": d.get("mark_iv"),
                "underlying": d.get("underlying_price"), "gamma": (d.get("greeks") or {}).get("gamma"),
                "delivery": d.get("estimated_delivery_price"),
            })
        except Exception as e:
            rows.append({"instrument": n, "expiry": i["expiry_ts"], "err": str(e)[:50]})

df = pd.DataFrame(rows)
print("=" * 92)
print("TASK 2 -- terminal open interest on expired options via get_order_book")
print("=" * 92)
print(df.to_string(index=False, max_colwidth=26))

ok = df[df.get("open_interest").notna()] if "open_interest" in df else df.iloc[:0]
print(f"\n  responded            : {len(ok)}/{len(df)}")
if len(ok):
    print(f"  open_interest non-null: {int(ok['open_interest'].notna().sum())}/{len(ok)}")
    print(f"  open_interest > 0     : {int((ok['open_interest'] > 0).sum())}/{len(ok)}")
    print(f"  OI range              : {ok['open_interest'].min():,.1f} .. {ok['open_interest'].max():,.1f}")
    print(f"  gamma all zero        : {bool((ok['gamma'].fillna(0) == 0).all())}")
    print(f"  settlement populated  : {int(ok['settlement'].notna().sum())}/{len(ok)}")

# --- plausibility: sum OI across every strike of ONE expiry ---
EXP = "2024-03-29"
same = inst[(inst["kind"] == "option") &
            (inst["expiry_ts"].dt.strftime("%Y-%m-%d") == EXP)]
print(f"\n  --- plausibility: all {len(same)} strikes of the {EXP} quarterly expiry ---")
tot, got, per = 0.0, 0, []
with httpx.Client(headers={"User-Agent": "Mozilla/5.0 crypto-gamma"}, timeout=40) as c:
    for n in same["instrument_name"]:
        try:
            r = c.get(f"{BASE}/public/get_order_book",
                      params={"instrument_name": n, "depth": 1}).json()
            oi = r.get("result", {}).get("open_interest")
            if oi is not None:
                tot += oi; got += 1
                per.append({"instrument": n, "oi": oi,
                            "strike": same.loc[same["instrument_name"] == n, "strike"].iloc[0]})
        except Exception:
            pass
print(f"  instruments returning OI : {got}/{len(same)}")
print(f"  SUM terminal OI          : {tot:,.1f} BTC contracts")
if per:
    pdf = pd.DataFrame(per).sort_values("oi", ascending=False)
    print(f"  max-OI strike            : {pdf.iloc[0]['strike']:,.0f} ({pdf.iloc[0]['oi']:,.1f})")
    print("  top 5 strikes by terminal OI:")
    for _, r in pdf.head(5).iterrows():
        print(f"    {r['instrument']:<26}{r['oi']:>12,.1f}")
    out = config.path("paths.interim") / f"terminal_oi_{EXP}.parquet"
    pdf.to_parquet(out, index=False)
    print(f"  saved -> {out}")
