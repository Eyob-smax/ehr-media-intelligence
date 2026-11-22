"""Format-specific loaders.

Loaders do as little interpretation as possible: they find the fields in each
source format and hand back raw strings. All normalization and identity
resolution happens in the pipeline so every source is treated the same way.
"""

from __future__ import annotations

import csv
import io
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


_CANONICAL_CSV = ["mrn", "patient_name", "dob", "collected", "panel", "test", "value", "unit", "ref_range", "flag"]
_CSV_HEADER_ALIASES = {
    "patient id": "mrn", "patientid": "mrn", "medical record number": "mrn", "mrn": "mrn",
    "patient name": "patient_name", "name": "patient_name", "patient_name": "patient_name",
    "birth date": "dob", "birthdate": "dob", "date of birth": "dob", "dob": "dob",
    "result date": "collected", "resultdatetime": "collected", "result datetime": "collected",
    "collection date": "collected", "collected": "collected", "specimen date": "collected",
    "order name": "panel", "ordername": "panel", "panel": "panel", "battery": "panel",
    "analyte": "test", "test name": "test", "testname": "test", "component": "test", "test": "test",
    "result": "value", "result value": "value", "value": "value",
    "units": "unit", "unit": "unit", "uom": "unit",
    "reference range": "ref_range", "refrange": "ref_range", "ref range": "ref_range", "ref_range": "ref_range",
    "abn flag": "flag", "abnflag": "flag", "abnormal flag": "flag", "flag": "flag",
}


def _hkey(h: str) -> str:
    return " ".join(h.strip().lstrip("\ufeff").lower().replace("_", " ").split()) if h else ""


def _read_text(path: Path) -> tuple[str, str]:
    """Decode as UTF-8 (with or without BOM); fall back to Latin-1, which never fails, for legacy exports."""
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("latin-1"), "latin-1"


def load_lab_csv(path: Path) -> LoadResult:
    """Group analyte rows into one lab report per (patient, collection time, panel).

    Exact duplicate rows and repeated analytes within a report are dropped and
    logged. Rows with an empty value are kept only if no other row supplies a
    value for the same analyte.
    """
    out = LoadResult()
    src = path.name
    groups: dict[tuple, dict[str, Any]] = {}
    text, encoding = _read_text(path)
    if encoding != "utf-8":
        out.audit.append(AuditEntry(record_id=src, field="encoding", action="normalized", before=encoding,
                                    after="utf-8", reason="file was not UTF-8; decoded as " + encoding, source=src))
    # The header line has no quoted commas or decimal commas, so the most frequent candidate there is the delimiter.
    # (csv.Sniffer gives up on files mixing ';' delimiters with decimal commas.)
    first_line = text.split("\n", 1)[0]
    delimiter = max(",;\t|", key=first_line.count)
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    header = [_CSV_HEADER_ALIASES.get(_hkey(h), _hkey(h)) for h in next(reader, [])]
    missing = {"mrn", "test"} - set(header)
    if missing:
        out.audit.append(AuditEntry(record_id=src, field="header", action="dropped",
                                    reason=f"unrecognized lab CSV layout (missing {sorted(missing)})", source=src))
        return out
    if header != _CANONICAL_CSV:
        out.audit.append(AuditEntry(record_id=src, field="header", action="normalized", after=header,
                                    reason="vendor column names mapped to canonical lab schema", source=src))
    for line_no, cells in enumerate(reader, start=2):
        rid = f"{src}:{line_no}"
        if not any(c.strip() for c in cells):
            continue
        if len(cells) != len(header):
            out.audit.append(AuditEntry(record_id=rid, field="row", action="dropped",
                                        before="|".join(cells)[:80],
                                        reason=f"expected {len(header)} columns, found {len(cells)}", source=src))
            continue
        row = {k: (v or "").strip() for k, v in zip(header, cells)}
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


_IMAGING_TITLE = re.compile(r"\b(radiology|imaging|x-?ray|xr|ct|mri|ultrasound|us|sonogram|mammogram)\b")


def _kind_from_title(title: str) -> DocKind:
    t = title.lower()
    if re.search(r"\bdischarge\b", t):
        return DocKind.discharge_summary
    if _IMAGING_TITLE.search(t):  # whole words only: "Direct Admission" must not match "ct"
        return DocKind.imaging_report
    if re.search(r"\b(progress|office visit)\b", t):
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
