"""SQLite summary cache.

Key = sha256(patient_id | record_hash | prompt_version | model). The record
hash covers every Bundle entry, so any change to a patient's records, the
prompt, or the model produces a new key and a fresh summary; re-running the
pipeline on unchanged data makes zero API calls.
"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .schema import ClinicalSummary

SCHEMA = """
CREATE TABLE IF NOT EXISTS summaries (
    cache_key      TEXT PRIMARY KEY,
    patient_id     TEXT NOT NULL,
    record_hash    TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model          TEXT NOT NULL,
    summary_json   TEXT NOT NULL,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_summaries_patient ON summaries(patient_id, created_at);
"""


def cache_key(patient_id: str, record_hash: str, prompt_version: str, model: str) -> str:
    return hashlib.sha256(f"{patient_id}|{record_hash}|{prompt_version}|{model}".encode()).hexdigest()


class SummaryCache:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get(self, key: str) -> ClinicalSummary | None:
        with self._conn() as c:
            row = c.execute("SELECT summary_json FROM summaries WHERE cache_key = ?", (key,)).fetchone()
        return ClinicalSummary.model_validate_json(row[0]) if row else None

    def put(self, key: str, s: ClinicalSummary) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO summaries VALUES (?,?,?,?,?,?,?)",
                (key, s.patient_id, s.record_hash, s.provenance.prompt_version, s.provenance.model or "",
                 s.model_dump_json(), datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )

    def latest_for(self, patient_id: str, record_hash: str | None = None) -> ClinicalSummary | None:
        """Most recent summary for a patient (optionally only for the current record hash)."""
        q = "SELECT summary_json FROM summaries WHERE patient_id = ?"
        args: tuple = (patient_id,)
        if record_hash:
            q += " AND record_hash = ?"
            args += (record_hash,)
        with self._conn() as c:
            row = c.execute(q + " ORDER BY created_at DESC LIMIT 1", args).fetchone()
        return ClinicalSummary.model_validate_json(row[0]) if row else None
