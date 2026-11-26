"""Append-only record of every specification the study has run.

WHY THIS FILE EXISTS. The Deflated Sharpe Ratio needs the number of distinct
specifications tried and the spread of their Sharpe ratios. Both are trivially
knowable while the study is running and impossible to reconstruct afterwards --
nobody remembers on Friday how many quintile schemes they tried on Tuesday, and
the ones that looked bad are exactly the ones that get forgotten. So each
analysis records itself as it runs, and the deflation reads the count from
here. The bar the headline result has to clear rises by itself, every run.

THE DEDUPLICATION RULE, which is the one real design decision here. Every run
appends a row, so the file is a complete audit trail with timestamps. But a
TRIAL is a distinct specification, not a distinct execution: re-running the
same analysis five times is one trial, not five, and counting it as five would
deflate the Sharpe for work never done. `trials()` therefore collapses on
`spec_id` -- a hash of the hypothesis, the name and the parameters -- keeping
the most recent row. Change any parameter and you get a new spec_id and a new
trial, which is correct: that is a new thing tried.

The config fingerprint on every row catches the other direction. Two rows with
the same spec_id and different `config_sha` were run under different
construction choices and are not comparable; `scripts/report_stats.py` flags it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src import config

log = logging.getLogger("utils.speclog")

COLUMNS = [
    "run_ts",        # UTC ISO timestamp of the run
    "run_id",        # one id per process, so a batch of specs can be grouped
    "spec_id",       # hash of (hypothesis, name, params) -- the trial identity
    "hypothesis",    # h1..h6, or a free label for exploratory work
    "name",          # short human name for the specification
    "params",        # JSON of the choices that define this specification
    "sample_start",
    "sample_end",
    "n_obs",
    "statistic",     # what `estimate` measures, e.g. "mean_realized_vol"
    "estimate",
    "se",            # HAC standard error where one applies
    "t_stat",
    "p_value",
    "sharpe",        # PER OBSERVATION -- see src/utils/stats.py convention
    "interval",      # the analysis grid the sharpe is measured on
    "config_sha",    # fingerprint of config.yaml at run time
    "notes",
]

# One id per interpreter process. Groups the specs a single run emitted.
RUN_ID = uuid.uuid4().hex[:12]


def log_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get("CRYPTO_GAMMA_SPEC_LOG")
    if env:
        return Path(env)
    return config.path("stats.spec_log")


def config_sha() -> str:
    """Short fingerprint of config.yaml, so rows from different settings are visible."""
    p = config.ROOT / "config.yaml"
    try:
        return hashlib.sha1(p.read_bytes()).hexdigest()[:10]
    except OSError:
        return "unknown"


def spec_id(hypothesis: str, name: str, params: dict[str, Any] | None) -> str:
    """Stable identity of a specification.

    Canonical JSON with sorted keys, so {"a":1,"b":2} and {"b":2,"a":1} are
    the same trial. Floats are formatted through repr, so 0.1 and 0.10 agree
    but 0.1 and 0.1000000001 do not -- which is right, they are different
    choices.
    """
    blob = json.dumps(
        {"hypothesis": hypothesis, "name": name, "params": params or {}},
        sort_keys=True, separators=(",", ":"), default=str,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def record(
    hypothesis: str,
    name: str,
    *,
    params: dict[str, Any] | None = None,
    n_obs: int | None = None,
    statistic: str = "",
    estimate: float | None = None,
    se: float | None = None,
    t_stat: float | None = None,
    p_value: float | None = None,
    sharpe: float | None = None,
    sample: tuple[str, str] | None = None,
    notes: str = "",
    path: str | Path | None = None,
) -> str:
    """Append one specification to the log. Returns its spec_id.

    Call this for EVERY specification run, including the ones that come out
    badly. A log containing only the winners deflates nothing and is worse
    than no log, because it looks like due diligence.
    """
    sid = spec_id(hypothesis, name, params)
    start, end = sample or (config.get("sample.start"), config.get("sample.end"))
    row = {
        "run_ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": RUN_ID,
        "spec_id": sid,
        "hypothesis": hypothesis,
        "name": name,
        "params": json.dumps(params or {}, sort_keys=True, default=str),
        "sample_start": start,
        "sample_end": end,
        "n_obs": n_obs,
        "statistic": statistic,
        "estimate": estimate,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "sharpe": sharpe,
        "interval": config.get("construction.interval"),
        "config_sha": config_sha(),
        "notes": notes,
    }

    p = log_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = not p.exists() or p.stat().st_size == 0
    pd.DataFrame([row], columns=COLUMNS).to_csv(p, mode="a", header=header, index=False)
    log.info("spec logged: %s/%s [%s]", hypothesis, name, sid)
    return sid


def load(path: str | Path | None = None) -> pd.DataFrame:
    """Every row ever written, in run order. Empty frame if nothing logged yet."""
    p = log_path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNS)
    # Identifier columns must be read as TEXT. A 10-hex-digit fingerprint such
    # as "3687797e33" is valid scientific notation, so pandas silently parses
    # it as 3.688e+39 and the row no longer matches the config it came from --
    # a fingerprint that compares unequal to itself defeats the whole point of
    # stamping one. Same hazard for spec_id and any hex-shaped id.
    id_cols = [c for c in ("spec_id", "config_sha", "hypothesis", "name",
                           "statistic", "sample_start", "sample_end", "notes")
               if c in COLUMNS]
    df = pd.read_csv(p, dtype={c: "string" for c in id_cols})
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[COLUMNS]


def trials(hypothesis: str | None = None, path: str | Path | None = None) -> pd.DataFrame:
    """Distinct specifications tried -- one row per spec_id, most recent run.

    This is the count the Deflated Sharpe Ratio consumes. See the module
    docstring for why it deduplicates on spec_id rather than counting rows.
    """
    df = load(path)
    if df.empty:
        return df
    if hypothesis is not None:
        df = df[df["hypothesis"] == hypothesis]
    if df.empty:
        return df
    return (df.sort_values("run_ts")
              .drop_duplicates("spec_id", keep="last")
              .reset_index(drop=True))


def summary(path: str | Path | None = None) -> pd.DataFrame:
    """Trials, runs and Sharpe spread per hypothesis. What the deflation sees."""
    df = load(path)
    if df.empty:
        return pd.DataFrame(columns=["hypothesis", "trials", "runs", "sharpes_logged",
                                     "sharpe_var", "sharpe_max", "last_run"])
    uniq = df.sort_values("run_ts").drop_duplicates("spec_id", keep="last")
    rows = []
    for h, g in uniq.groupby("hypothesis", dropna=False):
        sr = pd.to_numeric(g["sharpe"], errors="coerce").dropna()
        rows.append({
            "hypothesis": h,
            "trials": len(g),
            "runs": int(df[df["hypothesis"] == h]["run_id"].nunique()),
            "sharpes_logged": len(sr),
            "sharpe_var": float(sr.var(ddof=1)) if len(sr) > 1 else float("nan"),
            "sharpe_max": float(sr.max()) if len(sr) else float("nan"),
            "last_run": g["run_ts"].max(),
        })
    total = {
        "hypothesis": "ALL", "trials": len(uniq), "runs": int(df["run_id"].nunique()),
        "sharpes_logged": int(pd.to_numeric(uniq["sharpe"], errors="coerce").notna().sum()),
        "sharpe_var": float(pd.to_numeric(uniq["sharpe"], errors="coerce").dropna().var(ddof=1)),
        "sharpe_max": float(pd.to_numeric(uniq["sharpe"], errors="coerce").max()),
        "last_run": df["run_ts"].max(),
    }
    return pd.DataFrame([*rows, total])


def config_drift(path: str | Path | None = None) -> pd.DataFrame:
    """Specs whose logged rows span more than one config fingerprint.

    Those rows are not comparable to each other: something in the
    construction changed between them. Rerun the spec, or split the trial.
    """
    df = load(path)
    if df.empty:
        return pd.DataFrame(columns=["spec_id", "hypothesis", "name", "config_shas"])
    g = df.groupby(["spec_id", "hypothesis", "name"])["config_sha"].nunique()
    drifted = g[g > 1].reset_index().rename(columns={"config_sha": "config_shas"})
    return drifted
