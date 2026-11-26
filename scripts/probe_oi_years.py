"""Terminal-OI coverage by year: sample 45 strikes per quarterly expiry."""
import asyncio, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.fetch.client import DeribitClient

TARGETS = ["2022-03-25","2022-12-30","2023-03-31","2023-12-29",
           "2024-03-29","2024-12-27","2025-03-28","2025-12-26"]

async def main():
    inst = pd.read_parquet(config.path("paths.interim")/"instruments.parquet")
    o = inst[inst["kind"]=="option"].copy()
    o["ed"] = o["expiry_ts"].dt.strftime("%Y-%m-%d")
    print(f"{'expiry':<13}{'sampled':>8}{'resp':>6}{'nonzero':>9}{'mean OI':>11}{'max-OI K':>11}")
    print("-"*60)
    async with DeribitClient(host="www") as c:
        for ed in TARGETS:
            same = o[o["ed"]==ed]
            if not len(same):
                print(f"{ed:<13}  -- no instruments --"); continue
            s = same.sample(min(45,len(same)), random_state=42)
            async def one(n):
                try:
                    r = await c.get("get_order_book", instrument_name=n, depth=1)
                    return r.get("open_interest")
                except Exception:
                    return None
            res = await asyncio.gather(*(one(n) for n in s["instrument_name"]))
            vals = [v for v in res if v is not None]
            nz = [v for v in vals if v > 0]
            best = max(zip(res, s["strike"]), key=lambda t: (t[0] or -1))
            print(f"{ed:<13}{len(s):>8}{len(vals):>6}{len(nz):>9}"
                  f"{(sum(vals)/len(vals) if vals else 0):>11,.1f}"
                  f"{(f'{best[1]:,.0f}' if best[0] else '-'):>11}")

asyncio.run(main())
