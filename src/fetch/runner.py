"""Shared machinery for day-partitioned, resumable backfills.

One UTC day is the atomic unit of work: a day is fetched in full, written to a
temporary file, fsynced, atomically renamed into place, and only THEN recorded
as `done`. An interrupted run therefore never records a partial day, and
resuming simply re-runs whatever is not marked done.

On SIGINT/SIGTERM we stop handing out new days and let in-flight days finish
(seconds each), so the checkpoint database and the raw files always agree.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import os
import signal
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tqdm import tqdm

from src import config
from src.fetch.checkpoint import Checkpoints
from src.fetch.client import DeribitClient

log = logging.getLogger("fetch.runner")

DAY_MS = 86_400_000

# Returns (trades, requests_used) for one [start_ms, end_ms) window.
PageFn = Callable[[DeribitClient, int, int], Awaitable[tuple[list[dict], int]]]


def day_range(start_ms: int, end_ms: int) -> list[tuple[str, int, int]]:
    """Split [start_ms, end_ms) into (key, start, end) UTC days."""
    out = []
    cur = start_ms - (start_ms % DAY_MS)
    while cur < end_ms:
        key = datetime.fromtimestamp(cur / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        out.append((key, max(cur, start_ms), min(cur + DAY_MS, end_ms)))
        cur += DAY_MS
    return out


async def paginate_window(
    client: DeribitClient,
    method: str,
    start_ms: int,
    end_ms: int,
    page_size: int,
    extra: dict,
    max_pages: int = 2000,
) -> tuple[list[dict], int]:
    """Page a time-windowed trades endpoint until the window is exhausted.

    Cursor advances to max(timestamp) -- NOT max+1 -- because several trades
    can share a millisecond and +1 would silently drop the rest of that
    millisecond. The resulting boundary duplicates are removed at parquet
    export on (instrument_name, trade_seq), which is exactly the behaviour the
    RiveChen docs describe.

    `has_more` is deliberately not trusted as a completeness signal (their docs
    are explicit that it is not a completeness proof); we page until a short
    page is returned or the cursor passes the window end.
    """
    trades: list[dict] = []
    cursor = start_ms
    requests = 0
    for _ in range(max_pages):
        res = await client.get(
            method,
            start_timestamp=cursor,
            end_timestamp=end_ms,
            count=page_size,
            sorting="asc",
            **extra,
        )
        requests += 1
        page = res.get("trades", [])
        if not page:
            break
        trades.extend(page)
        if len(page) < page_size:
            break
        mx = max(t["timestamp"] for t in page)
        cursor = mx if mx > cursor else cursor + 1
        if cursor >= end_ms:
            break
    else:
        log.warning("max_pages hit for window %s..%s", start_ms, end_ms)
    return trades, requests


def _write_day(path: Path, rows: list[dict], compress: bool) -> int:
    """Atomically write one day of raw JSONL. Returns bytes written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)
    raw = payload.encode()
    if compress:
        with open(tmp, "wb") as fh:
            with gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:
                gz.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
    else:
        with open(tmp, "wb") as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
    os.replace(tmp, path)          # atomic within the same filesystem
    return path.stat().st_size


async def run_daily_backfill(
    stage: str,
    page_fn: PageFn,
    out_dir: Path,
    start_ms: int,
    end_ms: int,
    host: str = "history",
) -> dict:
    """Fetch [start_ms, end_ms) one UTC day at a time, resumable throughout."""
    cp = Checkpoints(config.path("fetch.storage.checkpoint_db"))
    compress = config.get("fetch.storage.compress", True)
    workers = config.get("fetch.workers", 8)
    ext = ".jsonl.gz" if compress else ".jsonl"

    all_days = day_range(start_ms, end_ms)
    done = cp.done_keys(stage)
    todo = [d for d in all_days if d[0] not in done]
    log.info("%s: %d days total, %d already done, %d to fetch",
             stage, len(all_days), len(done), len(todo))
    if not todo:
        log.info("%s: nothing to do", stage)
        cp.close()
        return {"stage": stage, "fetched_days": 0, "rows": 0}

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_signal(signame: str) -> None:
        if not stop.is_set():
            log.warning("%s received -- finishing in-flight days, then exiting cleanly", signame)
            stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal, sig.name)
        except NotImplementedError:      # pragma: no cover (non-unix)
            pass

    queue: asyncio.Queue = asyncio.Queue()
    for task in todo:
        queue.put_nowait(task)

    totals = {"rows": 0, "requests": 0, "bytes": 0, "days": 0, "failed": 0}
    bar = tqdm(total=len(todo), desc=f"{stage}", unit="day", dynamic_ncols=True)
    t0 = time.time()

    async with DeribitClient(host=host) as client:

        async def worker(wid: int) -> None:
            while not stop.is_set():
                try:
                    key, s, e = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    rows, reqs = await page_fn(client, s, e)
                    nbytes = _write_day(out_dir / f"{key}{ext}", rows, compress)
                    cp.mark(stage, key, "done", rows=len(rows), requests=reqs, nbytes=nbytes)
                    totals["rows"] += len(rows)
                    totals["requests"] += reqs
                    totals["bytes"] += nbytes
                    totals["days"] += 1
                except Exception as exc:                     # noqa: BLE001
                    totals["failed"] += 1
                    log.error("%s %s FAILED: %s", stage, key, exc)
                    cp.mark(stage, key, "failed", error=str(exc)[:500])
                finally:
                    queue.task_done()
                    bar.update(1)
                    bar.set_postfix(rows=f"{totals['rows']:,}", failed=totals["failed"])

        await asyncio.gather(*(worker(i) for i in range(workers)))

    bar.close()
    elapsed = time.time() - t0
    log.info(
        "%s finished: %d days, %d rows, %d requests, %.2f GB, %.1f min (client retries=%d)",
        stage, totals["days"], totals["rows"], totals["requests"],
        totals["bytes"] / 1e9, elapsed / 60, client.n_retries,
    )
    if totals["failed"]:
        log.error("%s: %d days FAILED -- rerun the same command to retry them",
                  stage, totals["failed"])
    if stop.is_set():
        log.warning("%s: stopped early by signal; %d days remain. Rerun to resume.",
                    stage, queue.qsize())
    cp.close()
    return {"stage": stage, "elapsed_sec": elapsed, **totals}
