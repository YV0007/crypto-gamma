"""SQLite checkpoint store making every backfill fully resumable.

One row per (stage, key) task -- a task is one UTC day of one stage. A task is
only marked `done` AFTER its raw file has been fsynced and atomically renamed
into place, so an interrupted run can never leave a half-written file recorded
as complete.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    stage       TEXT NOT NULL,
    key         TEXT NOT NULL,
    status      TEXT NOT NULL,          -- pending | done | failed
    rows        INTEGER DEFAULT 0,
    requests    INTEGER DEFAULT 0,
    bytes       INTEGER DEFAULT 0,
    error       TEXT,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (stage, key)
);
CREATE INDEX IF NOT EXISTS idx_stage_status ON checkpoints(stage, status);
"""


class Checkpoints:
    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def done_keys(self, stage: str) -> set[str]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT key FROM checkpoints WHERE stage=? AND status='done'", (stage,)
            )
            return {r[0] for r in cur.fetchall()}

    def mark(
        self,
        stage: str,
        key: str,
        status: str,
        rows: int = 0,
        requests: int = 0,
        nbytes: int = 0,
        error: str | None = None,
    ) -> None:
        now = datetime.now(tz=timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """INSERT INTO checkpoints(stage,key,status,rows,requests,bytes,error,updated_at)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(stage,key) DO UPDATE SET
                     status=excluded.status, rows=excluded.rows,
                     requests=excluded.requests, bytes=excluded.bytes,
                     error=excluded.error, updated_at=excluded.updated_at""",
                (stage, key, status, rows, requests, nbytes, error, now),
            )
            self._conn.commit()

    def summary(self, stage: str) -> dict:
        with self._lock:
            cur = self._conn.execute(
                """SELECT status, COUNT(*), COALESCE(SUM(rows),0), COALESCE(SUM(requests),0),
                          COALESCE(SUM(bytes),0)
                   FROM checkpoints WHERE stage=? GROUP BY status""",
                (stage,),
            )
            out = {}
            for status, n, rows, reqs, nbytes in cur.fetchall():
                out[status] = {"tasks": n, "rows": rows, "requests": reqs, "bytes": nbytes}
            return out

    def close(self) -> None:
        with self._lock:
            self._conn.close()
