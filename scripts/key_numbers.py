"""Every number that may appear on a slide, pulled live from outputs/.

Numbers in this project have changed two or three times as construction faults
were found. Anything typed into a deck from memory, or from an old chat message,
has a real chance of being wrong. This script is the single source of truth:
run it, read the figure off the output, and never transcribe from anywhere else.
"""
from __future__ import annotations
import json
import pandas as pd
from src import config

TAB = config.path("paths.tables")
L = lambda f: json.loads((TAB / f).read_text())


def main() -> None:
    st, h3 = L("pipeline_stats.json"), L("h3_identification.json")
    val, h4 = L("gamma_aggregate_validation.json"), L("h4_results.json")
    h2w, rev = L("h2_within_day.json"), L("h3_reversal_diagnosis.json")
    h6 = L("h6_results.json")
    mt, ic = pd.read_csv(TAB / "multiple_testing.csv"), pd.read_csv(TAB / "signal_ic.csv")
    h2 = pd.read_csv(TAB / "h2_extreme_contrast.csv")
    base = config.get("h2_volatility.detrend.baseline")
    h2b = h2[h2["detrend"] == base].iloc[0]

    def sec(t): print(f"\n{'='*66}\n{t}\n{'='*66}")

    sec("SAMPLE")
    print(f"  option trades                 {st['n_trades_in']:>15,}")
    print(f"  instruments                   {st['instruments']:>15,}")
    print(f"  panel intervals               {st['intervals']:>15,}")
    print(f"  block share of volume         {100*st['block_fraction_volume']:>14.1f}%")
    print(f"  block share of net dealer flow{100*st['block_fraction_net_flow']:>14.1f}%")
    print(f"  position-weighted gamma cover {100*st['pos_weight_with_gamma']/st['pos_weight_total']:>14.1f}%")

    sec("VALIDATION  (slide 5)")
    a = val["aggregate"]
    print(f"  per-instrument correlation    {val['per_instrument']['corr_levels']:>15.4f}")
    print(f"  aggregate signed ratio        {a['signed_ratio']:>15.4f}")
    print(f"  aggregate gross ratio         {a['gross_ratio']:>15.4f}")

    sec("H2  volatility  (slide 9)")
    print(f"  top-minus-bottom quintile     {h2b['point']:>+15.6f}  [{h2b['boot_lo']:+.6f}, {h2b['boot_hi']:+.6f}]")
    for k, lab in (("baseline", "within-day beta"), ("excl_final_expiry_day", "  excl. near-expiry")):
        print(f"  {lab:<29} {h2w[k]['beta']:>+15.6f}  t {h2w[k]['t']:+.2f}")

    sec("H3  hedging channel  (slide 10)")
    for h in ("ofi_5m", "ofi_30m"):
        p, w = h3[h]["pooled"], h3[h]["within_day"]
        print(f"  {h:<8} pooled {p['did']:+.5f} (t {p['t']:+.2f})   within-day {w['did']:+.5f} (t {w['t']:+.2f})")
    for e in ("post_etf", "pre_etf"):
        x = h3["by_era"][e]["ofi_5m"]["pooled"]
        print(f"  {e:<10} 5m    {x['did']:+.5f}  t {x['t']:+.2f}")
    print("  -- the reversal is 2022 --")
    for k in ("y2022_only", "y2023_only", "pre_etf_excl_2022"):
        v = rev["evidence"][k]
        print(f"    {k:<20}{v['did']:>+11.5f}  t {v['t']:>+6.2f}")

    sec("H4  structural break  (slide 12)")
    b = h4["h2_beta_break"]
    print(f"  beta pre                      {b['pre']['coef']:>+15.6f}  p {b['pre']['p_value']:.4f}")
    print(f"  beta post                     {b['post']['coef']:>+15.6f}  p {b['post']['p_value']:.4f}")
    print(f"  difference                    {b['difference']['coef']:>+15.6f}  p {b['difference']['p_value']:.4f}")

    sec("H6 + PREDICTION  (slide 14)")
    bl = h6["baseline"]
    print(f"  net Sharpe (annualised)       {bl['sharpe_net_annual']:>15.2f}")
    print(f"  gross Sharpe (annualised)     {bl['sharpe_gross_annual']:>15.2f}")
    print(f"  breakeven cost (bps)          {bl['breakeven_cost_bps']:>15.2f}   actual {h6['cost_bps_per_unit_turnover']}")
    for _, r in ic[ic["pair"].str.startswith("signal -> fwd_rv")].iterrows():
        print(f"  IC {r['pair'].split('_')[-1]:<5} raw {r['ic_raw']:+.4f}   incremental {r['ic_incremental']:+.4f}"
              f"  [{r['inc_lo']:+.4f}, {r['inc_hi']:+.4f}]")

    sec("MULTIPLE TESTING  (slide 15)")
    print(f"  {int(mt['reject'].sum())} of {len(mt)} primary tests survive Benjamini-Hochberg at 5% FDR")
    for _, r in mt.sort_values("q_value").iterrows():
        print(f"    {'PASS' if r['reject'] else '    '}  q={r['q_value']:.4f}  {r['hypothesis']} · {r['test']}")
    print()


if __name__ == "__main__":
    main()
