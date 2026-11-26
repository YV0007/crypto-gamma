"""Follow-up recon: can historical option trades be reached WITHOUT
enumerating expired instruments?

Tests, in order of preference:
  1. get_instruments with start_timestamp/end_timestamp (expired window)
  2. get_last_trades_by_currency_and_time  <- enumeration-free, time-windowed
  3. get_tradingview_chart_data / get_volatility_index_data reach-back
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

import httpx

BASE = "https://www.deribit.com/api/v2"
HISTORY = "https://history.deribit.com/api/v2"


def ms(y, m, d) -> int:
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp() * 1000)


def ts(v) -> str:
    return datetime.fromtimestamp(v / 1000, tz=timezone.utc).isoformat()


def call(c, host, method, **params):
    r = c.get(f"{host}/public/{method}", params=params, timeout=60)
    payload = r.json()
    if "error" in payload:
        return {"__error__": payload["error"]}
    return payload["result"]


with httpx.Client(headers={"User-Agent": "crypto-gamma-recon/0.1"}) as c:
    print("=" * 70)
    print("1. get_instruments with expired window (2024-03)")
    print("=" * 70)
    for host, label in ((BASE, "www"), (HISTORY, "history")):
        res = call(c, host, "get_instruments", currency="BTC", kind="option",
                   expired="true", start_timestamp=ms(2024, 3, 1), end_timestamp=ms(2024, 4, 1))
        if "__error__" in res:
            print(f"  [{label}] ERROR: {res['__error__']}")
        else:
            print(f"  [{label}] returned {len(res)} instruments")
            if res:
                exp = sorted(i["expiration_timestamp"] for i in res)
                print(f"           expiry range {ts(exp[0])} .. {ts(exp[-1])}")
                print(f"           sample: {[i['instrument_name'] for i in res[:5]]}")

    print()
    print("=" * 70)
    print("2. get_last_trades_by_currency_and_time  (enumeration-free)")
    print("=" * 70)
    for (y, m, d), label in (((2024, 3, 1), "2024-03-01"),
                             ((2022, 1, 1), "2022-01-01"),
                             ((2021, 1, 1), "2021-01-01"),
                             ((2020, 1, 1), "2020-01-01")):
        start = ms(y, m, d)
        end = start + 3600_000  # one hour
        res = call(c, HISTORY, "get_last_trades_by_currency_and_time",
                   currency="BTC", kind="option", start_timestamp=start,
                   end_timestamp=end, count=10000, sorting="asc")
        if "__error__" in res:
            print(f"  {label}: ERROR {res['__error__']}")
            continue
        tr = res.get("trades", [])
        print(f"  {label}: {len(tr):>5} trades in 1h  has_more={res.get('has_more')}")
        if tr:
            print(f"           first ts {ts(tr[0]['timestamp'])}  last ts {ts(tr[-1]['timestamp'])}")
            print(f"           instruments seen: {len({t['instrument_name'] for t in tr})}")
            if label == "2024-03-01":
                print(f"           FIELDS: {sorted(tr[0].keys())}")
                dirs = Counter(t.get("direction") for t in tr)
                nblock = sum(1 for t in tr if t.get("block_trade_id"))
                print(f"           direction dist: {dict(dirs)}")
                print(f"           rows w/ block_trade_id: {nblock}/{len(tr)}")
                print(f"           has iv field: {'iv' in tr[0]}")
                print(f"           SAMPLE ROW:\n{json.dumps(tr[0], indent=14)}")
                bpt = len(json.dumps(tr[0]).encode())
                print(f"           bytes/trade: {bpt}")

    print()
    print("  --- daily volume probe (for sizing) ---")
    tot = 0
    start = ms(2024, 3, 5)
    cursor = start
    end = start + 86_400_000
    pages = 0
    while cursor < end and pages < 200:
        res = call(c, HISTORY, "get_last_trades_by_currency_and_time",
                   currency="BTC", kind="option", start_timestamp=cursor,
                   end_timestamp=end, count=10000, sorting="asc")
        tr = res.get("trades", [])
        if not tr:
            break
        tot += len(tr)
        newc = max(t["timestamp"] for t in tr) + 1
        if newc <= cursor:
            break
        cursor = newc
        pages += 1
        if not res.get("has_more") and len(tr) < 10000:
            break
    print(f"  2024-03-05 full-day option trades: {tot:,} in {pages} requests")

    print()
    print("=" * 70)
    print("3. market data reach-back")
    print("=" * 70)
    res = call(c, BASE, "get_tradingview_chart_data", instrument_name="BTC-PERPETUAL",
               start_timestamp=ms(2022, 1, 1), end_timestamp=ms(2022, 1, 2), resolution="1")
    if "__error__" in res:
        print(f"  ohlcv 2022-01-01: ERROR {res['__error__']}")
    else:
        print(f"  ohlcv 2022-01-01: status={res.get('status')} bars={len(res.get('ticks', []))}")
        if res.get("ticks"):
            print(f"                    first {ts(res['ticks'][0])} close={res['close'][0]}")
    res = call(c, BASE, "get_volatility_index_data", currency="BTC",
               start_timestamp=ms(2022, 1, 1), end_timestamp=ms(2022, 1, 3), resolution="60")
    if "__error__" in res:
        print(f"  dvol  2022-01-01: ERROR {res['__error__']}")
    else:
        d = res.get("data", [])
        print(f"  dvol  2022-01-01: {len(d)} points; first {ts(d[0][0]) if d else 'n/a'}")

    print()
    print("  --- perp trades reach-back ---")
    res = call(c, HISTORY, "get_last_trades_by_instrument_and_time",
               instrument_name="BTC-PERPETUAL", start_timestamp=ms(2022, 1, 1),
               end_timestamp=ms(2022, 1, 1) + 600_000, count=10000, sorting="asc")
    if "__error__" in res:
        print(f"  perp 2022-01-01: ERROR {res['__error__']}")
    else:
        tr = res.get("trades", [])
        print(f"  perp 2022-01-01: {len(tr)} trades in 10min, has_more={res.get('has_more')}")
        if tr:
            print(f"                   fields: {sorted(tr[0].keys())}")
            print(f"                   sample: {json.dumps(tr[0])[:300]}")
