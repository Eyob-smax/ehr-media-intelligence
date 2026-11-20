"""Format-specific loaders.

Loaders do as little interpretation as possible: they find the fields in each
source format and hand back raw strings. All normalization and identity
resolution happens in the pipeline so every source is treated the same way.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .models import AuditEntry, DocKind
from .normalize import normalize_mrn, parse_datetime

# Source precedence when sources disagree: the EHR export is the system of record.
SOURCE_RANK = {"json": 0, "txt": 1, "csv": 2}


@dataclass
class RawIdentity:
    source_format: str
    record_id: str
    mrn: str | None = None
    name: str | None = None
    given: str | None = None
    family: str | None = None
    dob: str | None = None
    gender: str | None = None


@dataclass
class RawEncounter:
    record_id: str
    encounter_id: str
    identity: RawIdentity
    start: str | None
    encounter_class: str | None
    reason: str | None


@dataclass
class RawDocument:
    record_id: str
    source_format: str
    identity: RawIdentity
    kind: DocKind
    title: str
    date: str | None
    text: str
    encounter_id: str | None = None
    observations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LoadResult:
    encounters: list[RawEncounter] = field(default_factory=list)
    documents: list[RawDocument] = field(default_factory=list)
    audit: list[AuditEntry] = field(default_factory=list)

    def extend(self, other: "LoadResult") -> None:
        self.encounters += other.encounters
        self.documents += other.documents
        self.audit += other.audit


def _first(d: dict, *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


# ---------------------------------------------------------------------------
# JSON (simulated EHR export)
# ---------------------------------------------------------------------------

_DOC_KIND_ALIASES = {
    "discharge_summary": DocKind.discharge_summary,
    "discharge": DocKind.discharge_summary,
    "progress_note": DocKind.progress_note,
    "office_note": DocKind.progress_note,
    "ed_note": DocKind.ed_note,
    "emergency_note": DocKind.ed_note,
    "imaging_report": DocKind.imaging_report,
    "radiology_report": DocKind.imaging_report,
    "imaging": DocKind.imaging_report,
    "lab_report": DocKind.lab_report,
}


def load_json_export(path: Path) -> LoadResult:
    data = json.loads(path.read_text())
    records = data["records"] if isinstance(data, dict) else data
    out = LoadResult()
    src = path.name
    for i, rec in enumerate(records):
        rid = str(rec.get("record_id") or f"{src}#{i}")
        p = rec.get("patient") or {}
        ident = RawIdentity(
            source_format="json", record_id=rid,
            mrn=_first(p, "mrn", "MRN", "medical_record_number", "patient_id"),
            name=_first(p, "name", "full_name"),
            given=_first(p, "first_name", "given_name", "given"),
            family=_first(p, "last_name", "family_name", "family", "surname"),
            dob=_first(p, "dob", "DOB", "birthDate", "date_of_birth", "birth_date"),
            gender=_first(p, "gender", "sex", "Gender", "Sex"),
        )
        enc = rec.get("encounter") or {}
        enc_id = _first(enc, "id", "encounter_id")
        if enc_id:
            out.encounters.append(RawEncounter(
                record_id=rid, encounter_id=str(enc_id), identity=ident,
                start=_first(enc, "date", "admit_date", "start", "period_start"),
                encounter_class=_first(enc, "class", "type", "encounter_class"),
                reason=_first(enc, "reason", "chief_complaint"),
            ))
        for j, doc in enumerate(rec.get("documents") or []):
            raw_kind = str(_first(doc, "doc_type", "type", "kind") or "").lower()
            kind = _DOC_KIND_ALIASES.get(raw_kind)
            if kind is None:
                out.audit.append(AuditEntry(record_id=f"{rid}/doc{j}", field="doc_type", action="normalized",
                                            before=raw_kind, after=DocKind.scanned_document.value,
                                            reason="unknown document type; treated as generic document", source=src))
                kind = DocKind.scanned_document
            text = doc.get("text") or ""
            if not text.strip():
                out.audit.append(AuditEntry(record_id=f"{rid}/doc{j}", field="text", action="dropped",
                                            reason="document has no text", source=src))
                continue
            out.documents.append(RawDocument(
                record_id=f"{rid}/doc{j}", source_format="json", identity=ident, kind=kind,
                title=str(_first(doc, "title", "description") or kind.value.replace("_", " ").title()),
                date=_first(doc, "date", "created", "authored", "performed", "effective"),
                text=text, encounter_id=str(enc_id) if enc_id else None,
            ))
    return out


# ---------------------------------------------------------------------------
# CSV (lab results, one row per analyte)
# ---------------------------------------------------------------------------


def load_lab_csv(path: Path) -> LoadResult:
    """Group analyte rows into one lab report per (patient, collection time, panel).

    Exact duplicate rows and repeated analytes within a report are dropped and
    logged. Rows with an empty value are kept only if no other row supplies a
    value for the same analyte.
    """
    out = LoadResult()
    src = path.name
    groups: dict[tuple, dict[str, Any]] = {}
    with path.open(newline="") as fh:
        for line_no, row in enumerate(csv.DictReader(fh), start=2):
            rid = f"{src}:{line_no}"
            row = {k: (v or "").strip() for k, v in row.items()}
            try:
                mrn_key = normalize_mrn(row.get("mrn"))
            except ValueError:
                mrn_key = row.get("mrn") or f"unknown:{line_no}"
            try:
                collected = parse_datetime(row.get("collected")).value
                day_key = collected.date().isoformat()
            except ValueError:
                collected, day_key = None, row.get("collected", "")
            key = (mrn_key, day_key, row.get("panel", "").upper())
            g = groups.setdefault(key, {
                "rows": {}, "first_line": line_no, "ident": RawIdentity(
                    source_format="csv", record_id=rid, mrn=row.get("mrn"),
                    name=row.get("patient_name") or None, dob=row.get("dob") or None),
                "collected_raw": row.get("collected"), "collected": collected, "panel": row.get("panel"),
            })
            # Prefer the most precise timestamp seen for this group (some rows only carry a date).
            if collected and (g["collected"] is None or (collected.time() != collected.min.time() and g["collected"].time() == collected.min.time())):
                g["collected"], g["collected_raw"] = collected, row.get("collected")
            if not g["ident"].name and row.get("patient_name"):
                g["ident"].name = row["patient_name"]
            test = row.get("test", "")
            existing = g["rows"].get(test)
            if existing is not None:
                if existing.get("value") or not row.get("value"):
                    out.audit.append(AuditEntry(record_id=rid, field="row", action="deduplicated",
                                                before=f"{test}={row.get('value')}", reason=f"repeat of {test} in same report",
                                                source=src))
                    continue
                out.audit.append(AuditEntry(record_id=rid, field="value", action="filled", before="",
                                            after=row.get("value"), reason=f"empty {test} value replaced by later row",
                                            source=src))
            g["rows"][test] = row

    for (mrn_key, day_key, panel_key), g in groups.items():
        rows = list(g["rows"].values())
        obs = [{"test": r["test"], "value": r.get("value"), "unit": r.get("unit"),
                "ref_range": r.get("ref_range"), "flag": r.get("flag")} for r in rows]
        panel = g["panel"] or panel_key
        out.documents.append(RawDocument(
            record_id=f"{src}:{g['first_line']}", source_format="csv", identity=g["ident"],
            kind=DocKind.lab_report, title=f"{panel} Results",
            date=g["collected"].isoformat() if g["collected"] else g["collected_raw"],
            text="", observations=obs,
        ))
    return out


# ---------------------------------------------------------------------------
# Plain text (scanned documents with a free-text header)
# ---------------------------------------------------------------------------

_HDR_PATIENT = re.compile(r"PATIENT:\s*(?P<name>.+?)\s{2,}MRN:\s*(?P<mrn>.+?)\s*$", re.MULTILINE)
_HDR_DOB = re.compile(r"DOB:\s*(?P<dob>.+?)(?:\s{2,}SEX:\s*(?P<sex>\S+))?\s*$", re.MULTILINE)
_HDR_DOS = re.compile(r"DATE OF SERVICE:\s*(?P<dos>.+?)\s*$", re.MULTILINE)
_HDR_TYPE = re.compile(r"DOCUMENT TYPE:\s*(?P<type>.+?)\s*$", re.MULTILINE)
_HDR_RULE = re.compile(r"^-{10,}\s*$", re.MULTILINE)


def _kind_from_title(title: str) -> DocKind:
    t = title.lower()
    if "discharge" in t:
        return DocKind.discharge_summary
    if any(w in t for w in ("radiology", "imaging", "x-ray", "xr ", "ct ", "mri", "ultrasound")):
        return DocKind.imaging_report
    if "progress" in t or "office" in t:
        return DocKind.progress_note
    return DocKind.scanned_document


def load_scanned_text(path: Path) -> LoadResult:
    out = LoadResult()
    raw = path.read_text(errors="replace")
    rid = path.name
    rule = _HDR_RULE.search(raw)
    header, body = (raw[: rule.start()], raw[rule.end():]) if rule else (raw[:400], raw)
    pm, dm, sm, tm = (r.search(header) for r in (_HDR_PATIENT, _HDR_DOB, _HDR_DOS, _HDR_TYPE))
    if not pm:
        out.audit.append(AuditEntry(record_id=rid, field="header", action="dropped",
                                    reason="no PATIENT/MRN header found; cannot attribute document", source=rid))
        return out
    ident = RawIdentity(source_format="txt", record_id=rid, mrn=pm["mrn"], name=pm["name"],
                        dob=dm["dob"] if dm else None, gender=dm["sex"] if dm and dm["sex"] else None)
    title = tm["type"] if tm else "Scanned Document"
    out.documents.append(RawDocument(
        record_id=rid, source_format="txt", identity=ident, kind=_kind_from_title(title), title=title,
        date=sm["dos"] if sm else None, text=body,
    ))
    return out


# ---------------------------------------------------------------------------
# Directory loader
# ---------------------------------------------------------------------------


def load_directory(raw_dir: Path) -> LoadResult:
    """Load every supported file under ``raw_dir``. JSON first so it wins ties on duplicates."""
    out = LoadResult()
    files: Iterable[Path] = sorted(raw_dir.rglob("*"))
    by_ext: dict[str, list[Path]] = {".json": [], ".csv": [], ".txt": []}
    for f in files:
        if f.is_file() and f.suffix.lower() in by_ext:
            by_ext[f.suffix.lower()].append(f)
    for f in by_ext[".json"]:
        out.extend(load_json_export(f))
    for f in by_ext[".txt"]:
        out.extend(load_scanned_text(f))
    for f in by_ext[".csv"]:
        out.extend(load_lab_csv(f))
    return out
