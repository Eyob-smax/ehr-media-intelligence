"""SQLite-backed Bundle store.

One row per patient Bundle plus a denormalized ``resources`` table so the API
can list a patient's linked resources without parsing every Bundle. Bundles
are also written as plain JSON files for inspection and for loading into a
real FHIR server.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .text import bundle_patient, bundle_records, patient_display

SCHEMA = """
CREATE TABLE IF NOT EXISTS bundles (
    patient_id     TEXT PRIMARY KEY,
    bundle_id      TEXT NOT NULL,
    mrn            TEXT NOT NULL,
    patient_name   TEXT NOT NULL,
    bundle_json    TEXT NOT NULL,
    record_hash    TEXT NOT NULL,
    resource_count INTEGER NOT NULL,
    valid          INTEGER NOT NULL,
    issues_json    TEXT NOT NULL DEFAULT '[]',
    updated_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS resources (
    resource_id   TEXT PRIMARY KEY,
    patient_id    TEXT NOT NULL REFERENCES bundles(patient_id) ON DELETE CASCADE,
    resource_type TEXT NOT NULL,
    category      TEXT,
    title         TEXT,
    date          TEXT,
    conclusion    TEXT,
    text          TEXT
);
CREATE INDEX IF NOT EXISTS idx_resources_patient ON resources(patient_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_bundles_mrn ON bundles(mrn);
"""


def record_hash(bundle: dict[str, Any]) -> str:
    """Content hash of a Bundle's entries (canonical JSON), used as the summary cache key."""
    canonical = json.dumps(bundle.get("entry", []), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class BundleStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def save(self, patient_id: str, bundle: dict[str, Any], *, valid: bool = True,
             issues: list[dict] | None = None) -> str:
        name, mrn = patient_display(bundle_patient(bundle))
        h = record_hash(bundle)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._conn() as c:
            c.execute("DELETE FROM resources WHERE patient_id = ?", (patient_id,))
            c.execute(
                """INSERT INTO bundles (patient_id, bundle_id, mrn, patient_name, bundle_json, record_hash,
                                        resource_count, valid, issues_json, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(patient_id) DO UPDATE SET bundle_id=excluded.bundle_id, mrn=excluded.mrn,
                     patient_name=excluded.patient_name, bundle_json=excluded.bundle_json,
                     record_hash=excluded.record_hash, resource_count=excluded.resource_count,
                     valid=excluded.valid, issues_json=excluded.issues_json, updated_at=excluded.updated_at""",
                (patient_id, bundle["id"], mrn, name, json.dumps(bundle), h, len(bundle.get("entry", [])),
                 int(valid), json.dumps(issues or []), now),
            )
            c.executemany(
                "INSERT INTO resources VALUES (?,?,?,?,?,?,?,?)",
                [(r.resource_id, patient_id, r.resource_type, r.category, r.title, r.date, r.conclusion, r.text)
                 for r in bundle_records(bundle)],
            )
        return h

    def get_bundle(self, patient_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT bundle_json FROM bundles WHERE patient_id = ?", (patient_id,)).fetchone()
        return json.loads(row["bundle_json"]) if row else None

    def get_meta(self, patient_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("""SELECT patient_id, bundle_id, mrn, patient_name, record_hash, resource_count, valid,
                                      issues_json, updated_at FROM bundles WHERE patient_id = ?""",
                            (patient_id,)).fetchone()
        if not row:
            return None
        meta = dict(row)
        meta["valid"] = bool(meta["valid"])
        meta["issues"] = json.loads(meta.pop("issues_json"))
        return meta

    def list_patients(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("""SELECT patient_id, mrn, patient_name, resource_count, valid, record_hash
                                FROM bundles ORDER BY patient_name""").fetchall()
        return [dict(r) | {"valid": bool(r["valid"])} for r in rows]

    def iter_bundles(self) -> Iterator[tuple[str, dict[str, Any]]]:
        with self._conn() as c:
            rows = c.execute("SELECT patient_id, bundle_json FROM bundles ORDER BY patient_id").fetchall()
        for r in rows:
            yield r["patient_id"], json.loads(r["bundle_json"])

    def resources_for(self, patient_id: str) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("""SELECT resource_id, resource_type, category, title, date, conclusion, text
                                FROM resources WHERE patient_id = ? ORDER BY date DESC""", (patient_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_resource(self, resource_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT patient_id FROM resources WHERE resource_id = ?", (resource_id,)).fetchone()
        if not row:
            return None
        bundle = self.get_bundle(row["patient_id"]) or {}
        return next((e["resource"] for e in bundle.get("entry", []) if e["resource"]["id"] == resource_id), None)

    def count(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM bundles").fetchone()[0]


def write_bundle_files(bundles: dict[str, dict[str, Any]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for pid, b in bundles.items():
        p = out_dir / f"{pid}.json"
        p.write_text(json.dumps(b, indent=2))
        paths.append(p)
    return paths
