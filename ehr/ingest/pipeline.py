"""Cleaning pipeline: raw loader output -> audited canonical dataset.

Order of operations
1. Load every source file (JSON first, so the system of record wins ties).
2. Resolve identities: normalize each identity claim, group by canonical MRN,
   then merge MRN groups that share an exact name + birth date (catches typos
   such as transposed digits across systems). Field conflicts inside a patient
   are resolved by vote, with source precedence (json > txt > csv) breaking ties.
3. Build encounters (dropping re-sent duplicates by encounter id).
4. Build documents: normalize dates (falling back to the encounter date),
   clean OCR noise, attach lab observations, link labs to the nearest encounter,
   and drop duplicates by a whitespace/case-insensitive content fingerprint.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from .labs import convert, find_panel, find_test, parse_lab_value
from .loaders import SOURCE_RANK, LoadResult, RawIdentity, load_directory
from .models import (
    AuditEntry,
    CleanDataset,
    ClinicalDocument,
    DatasetIssue,
    DocKind,
    Encounter,
    Gender,
    LabObservation,
    Patient,
)
from .normalize import (
    clean_text,
    extract_suffix,
    derive_flag,
    fingerprint,
    mrn_to_patient_id,
    normalize_gender,
    normalize_mrn,
    parse_date,
    parse_datetime,
    split_name,
)

log = logging.getLogger(__name__)

LAB_ENCOUNTER_WINDOW = timedelta(days=2)



# ---------------------------------------------------------------------------
# Identity resolution
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    """One source's normalized statement about who a patient is."""

    raw: RawIdentity
    mrn: str | None
    given: str | None
    family: str | None
    birth_date: date | None
    gender: Gender | None
    suffix: str | None = None
    notes: list[tuple[str, str, object, object]] = field(default_factory=list)  # (field, note, before, after)

    @property
    def source_label(self) -> str:
        return f"{self.raw.source_format}:{self.raw.record_id}"


def _normalize_claim(raw: RawIdentity, reference: date, issues: list[DatasetIssue]) -> Claim:
    c = Claim(raw=raw, mrn=None, given=None, family=None, birth_date=None, gender=None)
    try:
        c.mrn = normalize_mrn(raw.mrn)
    except ValueError as e:
        issues.append(DatasetIssue(severity="warning", record_id=raw.record_id, message=str(e)))
    c.suffix = extract_suffix(raw.family or raw.name)[1]
    if raw.given or raw.family:
        g, f = split_name(f"{raw.family or ''}, {raw.given or ''}")
        c.given, c.family = g, f
    else:
        c.given, c.family = split_name(raw.name)
    if raw.dob:
        try:
            c.birth_date, note = parse_date(raw.dob, reference=reference, allow_future=False)
            if note:
                c.notes.append(("birth_date", note, raw.dob, c.birth_date))
        except ValueError as e:
            issues.append(DatasetIssue(severity="warning", record_id=raw.record_id, message=f"birth date: {e}"))
    if raw.gender is not None:
        c.gender, note = normalize_gender(raw.gender)
        if note and c.gender is Gender.unknown:
            c.notes.append(("gender", note, raw.gender, c.gender))
    return c


def _name_key(c: Claim) -> tuple | None:
    if not (c.family and c.given and c.birth_date):
        return None
    return (c.family.lower(), c.given.lower(), c.birth_date)


def _vote(values: list[tuple[object, Claim]]) -> tuple[object, Counter]:
    """Most frequent value; ties broken by best (lowest) source rank, then first seen."""
    counts: Counter = Counter(v for v, _ in values)
    best_rank: dict[object, int] = {}
    for v, c in values:
        best_rank[v] = min(best_rank.get(v, 99), SOURCE_RANK.get(c.raw.source_format, 9))
    winner = max(counts, key=lambda v: (counts[v], -best_rank[v]))
    return winner, counts


def _looks_like_keying_error(a: str, b: str) -> str | None:
    """Describe how two canonical MRNs differ if it looks like a typo, else None.

    Accepts one adjacent-digit transposition or up to two substituted digits,
    the two most common manual keying errors.
    """
    da, db = a.removeprefix("MRN-"), b.removeprefix("MRN-")
    if len(da) != len(db) or da == db:
        return None
    diffs = [i for i, (x, y) in enumerate(zip(da, db)) if x != y]
    if len(diffs) == 2 and diffs[1] == diffs[0] + 1 and da[diffs[0]] == db[diffs[1]] and da[diffs[1]] == db[diffs[0]]:
        return "an adjacent-digit transposition"
    if len(diffs) <= 2:
        return f"{len(diffs)} substituted digit{'s' if len(diffs) > 1 else ''}"
    return None


def _is_day_month_swap(a: date, b: date) -> bool:
    return a.year == b.year and a.month == b.day and a.day == b.month and a != b


class IdentityResolver:
    def __init__(self, reference: date, issues: list[DatasetIssue], dataset_audit: list[AuditEntry]):
        self.reference = reference
        self.issues = issues
        self.dataset_audit = dataset_audit
        self.claims: list[Claim] = []
        self._claim_by_raw: dict[int, Claim] = {}
        self.patients: dict[str, Patient] = {}          # patient_id -> Patient
        self._mrn_to_pid: dict[str, str] = {}
        self._claim_pid: dict[int, str] = {}           # id(RawIdentity) -> patient_id

    def add(self, raw: RawIdentity) -> None:
        if id(raw) in self._claim_by_raw:
            return
        c = _normalize_claim(raw, self.reference, self.issues)
        self.claims.append(c)
        self._claim_by_raw[id(raw)] = c

    def resolve(self) -> None:
        # 1. Group by canonical MRN.
        by_mrn: dict[str, list[Claim]] = defaultdict(list)
        orphans: list[Claim] = []
        for c in self.claims:
            (by_mrn[c.mrn] if c.mrn else orphans).append(c)

        # 2. Merge MRN groups that describe the same person (same name + DOB).
        group_key: dict[str, tuple | None] = {}
        for mrn, claims in by_mrn.items():
            keys = [_name_key(c) for c in claims if _name_key(c)]
            group_key[mrn] = Counter(keys).most_common(1)[0][0] if keys else None

        def group_rank(mrn: str) -> tuple:
            claims = by_mrn[mrn]
            return (min(SOURCE_RANK[c.raw.source_format] for c in claims), -len(claims), mrn)

        merged_into: dict[str, str] = {}
        by_key: dict[tuple, list[str]] = defaultdict(list)
        for mrn, key in group_key.items():
            if key:
                by_key[key].append(mrn)
        for key, mrns in by_key.items():
            if len(mrns) < 2:
                continue
            primary, *others = sorted(mrns, key=group_rank)
            for o in others:
                sources = sorted({c.raw.source_format for c in by_mrn[o]})
                typo = _looks_like_keying_error(primary, o)
                if "json" not in sources and typo:
                    merged_into[o] = primary
                    self.dataset_audit.append(AuditEntry(
                        record_id=f"Patient/{mrn_to_patient_id(primary)}", field="mrn", action="merged",
                        before=o, after=primary,
                        reason=(f"{o} (only in {sources}) has identical name and birth date to {primary} and "
                                f"differs by {typo}; treated as an identifier error and merged"),
                    ))
                else:
                    # Same name + DOB is not proof of identity (namesakes, twins with shared names, two real
                    # registrations). A wrong merge mixes two people's charts, so we flag for human review instead.
                    why = ("both MRNs are registered in the EHR export" if "json" in sources
                           else "MRNs differ by more than a typical keying error")
                    self.issues.append(DatasetIssue(
                        severity="warning", record_id=f"Patient/{mrn_to_patient_id(o)}",
                        message=f"possible duplicate of {primary} (same name and birth date) not merged: {why}",
                    ))

        # 3. Orphans (no usable MRN) attach by name + DOB if possible.
        key_to_mrn = {group_key[m]: m for m in by_mrn if group_key[m] and m not in merged_into}
        for c in orphans:
            k = _name_key(c)
            if k and k in key_to_mrn:
                by_mrn[key_to_mrn[k]].append(c)
                self.dataset_audit.append(AuditEntry(
                    record_id=c.raw.record_id, field="mrn", action="inferred", before=c.raw.mrn,
                    after=key_to_mrn[k], reason="invalid/missing MRN; matched to patient by name and birth date"))
            else:
                self.issues.append(DatasetIssue(severity="error", record_id=c.raw.record_id,
                                                message="no usable MRN and no name+DOB match; record dropped"))

        # 4. Build one Patient per surviving MRN group.
        for mrn in sorted(by_mrn, key=group_rank):
            if mrn in merged_into:
                continue
            claims = list(by_mrn[mrn])
            alternates = sorted(m for m, p in merged_into.items() if p == mrn)
            for alt in alternates:
                claims += by_mrn[alt]
            self._build_patient(mrn, claims, alternates)

    def _build_patient(self, mrn: str, claims: list[Claim], alternates: list[str]) -> None:
        pid = mrn_to_patient_id(mrn)
        p = Patient(patient_id=pid, mrn=mrn, alternate_mrns=alternates,
                    sources=sorted({c.source_label for c in claims}))

        raw_mrns = sorted({str(c.raw.mrn) for c in claims if c.raw.mrn and c.mrn == mrn and str(c.raw.mrn) != mrn})
        if raw_mrns:
            p.log("mrn", "normalized", before=raw_mrns, after=mrn, reason="MRN variants normalized to MRN-#######")

        # Names: vote on (given, family) pairs.
        names = [((c.given, c.family), c) for c in claims if c.given and c.family]
        if names:
            (p.given_name, p.family_name), counts = _vote(names)
            raw_forms = sorted({c.raw.name or f"{c.raw.given} {c.raw.family}" for _, c in names} - {f"{p.given_name} {p.family_name}"})
            if raw_forms:
                p.log("name", "normalized", before=raw_forms[:5], after=p.display_name,
                      reason="name parsed from 'FAMILY, Given' / 'Given Family' and title-cased")
            if len(counts) > 1:
                p.log("name", "conflict", before={f"{g} {f}": n for (g, f), n in counts.items()},
                      after=p.display_name, reason="sources disagree on name; majority vote")
        suffixes = [(c.suffix, c) for c in claims if c.suffix]
        if suffixes:
            p.name_suffix, _ = _vote(suffixes)
            p.log("name_suffix", "normalized", before=sorted({c.raw.family or c.raw.name for _, c in suffixes})[:3],
                  after=p.name_suffix, reason="generational suffix separated from family name")
        missing_name = [c for c in claims if not (c.given and c.family)]
        if missing_name and p.given_name:
            p.log("name", "filled", before=None, after=p.display_name,
                  reason=f"{len(missing_name)} source record(s) lacked a name; filled from identity match",
                  source=", ".join(sorted({c.source_label for c in missing_name})[:3]))

        # Birth date.
        dobs = [(c.birth_date, c) for c in claims if c.birth_date]
        if dobs:
            p.birth_date, counts = _vote(dobs)
            raw_forms = sorted({str(c.raw.dob) for _, c in dobs if str(c.raw.dob) != p.birth_date.isoformat()})
            if raw_forms:
                p.log("birth_date", "normalized", before=raw_forms[:6], after=p.birth_date,
                      reason="date formats normalized to ISO 8601")
            for c in claims:
                for fld, note, before, after in c.notes:
                    if fld == "birth_date" and after == p.birth_date:
                        p.log("birth_date", "inferred", before=before, after=after, reason=note, source=c.source_label)
                        break
            if len(counts) > 1:
                losers = [d for d in counts if d != p.birth_date]
                swap = any(_is_day_month_swap(d, p.birth_date) for d in losers)
                p.log("birth_date", "conflict", before={d.isoformat(): n for d, n in counts.items()},
                      after=p.birth_date,
                      reason=("day/month transposition suspected; " if swap else "") +
                             "kept the value reported by the most sources (EHR export breaks ties)")
        else:
            p.log("birth_date", "missing", reason="no source supplied a parseable birth date")

        # Gender.
        genders = [(c.gender, c) for c in claims if c.gender and c.gender is not Gender.unknown]
        if genders:
            p.gender, counts = _vote(genders)
            raw_forms = sorted({str(c.raw.gender) for _, c in genders if str(c.raw.gender) != p.gender.value})
            if raw_forms:
                p.log("gender", "normalized", before=raw_forms, after=p.gender,
                      reason="mapped to FHIR administrative-gender (numeric codes per ISO/IEC 5218)")
            if len(counts) > 1:
                p.log("gender", "conflict", before={g.value: n for g, n in counts.items()}, after=p.gender,
                      reason="sources disagree; majority vote")
        else:
            p.gender = Gender.unknown
            p.log("gender", "missing", after="unknown", reason="no source supplied a recognizable gender")

        self.patients[pid] = p
        self._mrn_to_pid[mrn] = pid
        for alt in alternates:
            self._mrn_to_pid[alt] = pid
        for c in claims:
            self._claim_pid[id(c.raw)] = pid

    def patient_id_for(self, raw: RawIdentity) -> str | None:
        return self._claim_pid.get(id(raw))


# ---------------------------------------------------------------------------
# Encounters and documents
# ---------------------------------------------------------------------------

_CLASS_MAP = {
    "inpatient": "inpatient", "ip": "inpatient", "admission": "inpatient", "imp": "inpatient",
    "outpatient": "outpatient", "op": "outpatient", "office": "outpatient", "amb": "outpatient", "ambulatory": "outpatient",
    "emergency": "emergency", "ed": "emergency", "er": "emergency", "emer": "emergency",
}


FLAG_WORDS = {"H": "high", "L": "low", "HH": "critical high", "LL": "critical low", "A": "abnormal"}


def _clean_observation(o: dict, doc: ClinicalDocument, audit_id: str) -> LabObservation:
    """Canonicalize one analyte: name/LOINC from the catalog, parse the value, convert units, derive the flag."""
    raw_name = str(o.get("test") or "").strip()
    test = find_test(raw_name)
    name = test.name if test else raw_name
    fld = f"observation:{name}"
    if test and raw_name != test.name:
        doc.log(fld, "normalized", before=raw_name, after=test.name, reason="test name alias mapped to catalog",
                record_id=audit_id)

    pv = parse_lab_value(o.get("value"))
    value, unit, ref = pv.value, (o.get("unit") or "").strip() or None, (o.get("ref_range") or "").strip() or None
    if pv.note:
        doc.log(fld, "normalized" if value is not None else "missing", before=o.get("value"), after=value,
                reason=pv.note, record_id=audit_id)
    elif value is None:
        doc.log(fld, "missing", before=o.get("value"), reason="lab value empty in source", record_id=audit_id)

    if test and value is not None:
        new_value, new_unit, note = convert(test, value, unit)
        if note and new_unit == test.unit and (unit or "") != test.unit:
            doc.log(fld, "normalized", before=f"{value:g} {unit}", after=f"{new_value:g} {new_unit}", reason=note,
                    record_id=audit_id)
            ref = test.ref_range or ref  # the source range was in the source unit
        elif note:
            doc.log(fld, "normalized", before=unit, reason=note, record_id=audit_id)
        value, unit = new_value, new_unit
    elif test and not unit:
        unit = test.unit or None
    if test and not ref and test.ref_range:
        ref = test.ref_range
        doc.log(fld, "filled", after=ref, reason="reference range missing; catalog default used", record_id=audit_id)

    flag, note = derive_flag(value, ref, pv.flag_hint or o.get("flag"))
    if note and value is not None:
        doc.log(fld, "inferred", before=o.get("flag"), after=flag, reason=note, record_id=audit_id)
    return LabObservation(test=name, value=value, comparator=pv.comparator, unit=unit or None,  # type: ignore[arg-type]
                          ref_range=ref, flag=flag, loinc=test.loinc if test else None)  # type: ignore[arg-type]


def _render_lab_text(title: str, when: datetime | None, obs: list[LabObservation]) -> str:
    lines = [title.upper(), f"Collected: {when:%Y-%m-%d %H:%M}" if when else "Collected: unknown"]
    for o in obs:
        val = "not reported" if o.value is None else f"{o.comparator or ''}{o.value:g} {o.unit or ''}".strip()
        flag = f" [{FLAG_WORDS[o.flag].upper()}]" if o.flag in FLAG_WORDS else ""
        ref = f" (ref {o.ref_range})" if o.ref_range else ""
        lines.append(f"{o.test}: {val}{ref}{flag}")
    abnormal = [o for o in obs if o.is_abnormal]
    lines.append("Abnormal results: " + (", ".join(f"{o.test} {FLAG_WORDS[o.flag]}" for o in abnormal)  # type: ignore[index]
                                         if abnormal else "none"))
    return "\n".join(lines)


def run_ingestion(raw_dir: Path, *, reference: date | None = None) -> CleanDataset:
    """Run the full Task 1 pipeline over every supported file in ``raw_dir``."""
    return clean(load_directory(raw_dir), reference=reference)


def clean(loaded: LoadResult, *, reference: date | None = None) -> CleanDataset:
    reference = reference or date.today()
    ds = CleanDataset(dataset_audit=list(loaded.audit))

    resolver = IdentityResolver(reference, ds.issues, ds.dataset_audit)
    for e in loaded.encounters:
        resolver.add(e.identity)
    for d in loaded.documents:
        resolver.add(d.identity)
    resolver.resolve()
    ds.patients = list(resolver.patients.values())

    # --- encounters ---------------------------------------------------------
    encounters: dict[str, Encounter] = {}
    for re_ in loaded.encounters:
        pid = resolver.patient_id_for(re_.identity)
        if pid is None:
            continue
        if re_.encounter_id in encounters:
            ds.dataset_audit.append(AuditEntry(record_id=re_.record_id, field="encounter", action="deduplicated",
                                               before=re_.encounter_id,
                                               reason="encounter id already ingested (re-sent export record)"))
            continue
        enc = Encounter(encounter_id=re_.encounter_id, patient_id=pid, sources=[f"json:{re_.record_id}"])
        if re_.start:
            try:
                parsed = parse_datetime(re_.start, reference=reference)
                enc.start = parsed.value
                if str(re_.start) != parsed.value.isoformat():
                    enc.log("start", "normalized", before=re_.start, after=parsed.value,
                            reason=parsed.note or "date normalized to ISO 8601")
            except ValueError as e:
                enc.log("start", "missing", before=re_.start, reason=str(e))
        cls = _CLASS_MAP.get(str(re_.encounter_class or "").strip().lower(), "unknown")
        if cls == "unknown":
            enc.log("encounter_class", "normalized", before=re_.encounter_class, after=cls,
                    reason="unrecognized encounter class")
        enc.encounter_class = cls  # type: ignore[assignment]
        enc.reason = (re_.reason or "").strip() or None
        encounters[enc.encounter_id] = enc
    ds.encounters = list(encounters.values())
    enc_by_patient: dict[str, list[Encounter]] = defaultdict(list)
    for e in ds.encounters:
        enc_by_patient[e.patient_id].append(e)

    # --- documents ----------------------------------------------------------
    seen: dict[str, ClinicalDocument] = {}
    for rd in loaded.documents:
        pid = resolver.patient_id_for(rd.identity)
        if pid is None:
            ds.dataset_audit.append(AuditEntry(record_id=rd.record_id, field="patient", action="dropped",
                                               reason="document could not be attributed to a patient"))
            continue
        doc = ClinicalDocument(
            doc_id="pending", patient_id=pid, kind=rd.kind, title=" ".join(rd.title.split()),
            text="", source_format=rd.source_format,  # type: ignore[arg-type]
            encounter_id=rd.encounter_id if rd.encounter_id in encounters else None,
            sources=[f"{rd.source_format}:{rd.record_id}"],
        )
        audit_id = f"{rd.source_format}:{rd.record_id}"

        # Date: parse, else fall back to the encounter start.
        if rd.date:
            try:
                parsed = parse_datetime(rd.date, reference=reference)
                doc.date = parsed.value
                if str(rd.date) != parsed.value.isoformat():
                    doc.log("date", "normalized", before=rd.date, after=parsed.value,
                            reason=parsed.note or "date normalized to ISO 8601", record_id=audit_id)
            except ValueError as e:
                doc.log("date", "dropped", before=rd.date, reason=str(e), record_id=audit_id)
        if doc.date is None and doc.encounter_id and encounters[doc.encounter_id].start:
            doc.date = encounters[doc.encounter_id].start
            doc.log("date", "filled", after=doc.date, reason="document date missing; used encounter start",
                    record_id=audit_id)
        if doc.date is None:
            doc.log("date", "missing", reason="no document or encounter date available", record_id=audit_id)

        # Lab observations.
        if rd.kind is DocKind.lab_report:
            panel = find_panel(rd.title.removesuffix(" Results"))
            if panel and f"{panel.name} Results" != doc.title:
                doc.log("title", "normalized", before=doc.title, after=f"{panel.name} Results",
                        reason="panel name mapped to catalog", record_id=audit_id)
                doc.title = f"{panel.name} Results"
            for o in rd.observations:
                doc.observations.append(_clean_observation(o, doc, audit_id))
            doc.text = _render_lab_text(doc.title, doc.date, doc.observations)
            # Labs arrive without an encounter; attach to the patient's nearest encounter within the window.
            if doc.date and not doc.encounter_id:
                near = [e for e in enc_by_patient[pid] if e.start and abs(e.start - doc.date) <= LAB_ENCOUNTER_WINDOW]
                if near:
                    best = min(near, key=lambda e: abs(e.start - doc.date))  # type: ignore[operator]
                    doc.encounter_id = best.encounter_id
                    doc.log("encounter_id", "inferred", after=best.encounter_id,
                            reason=f"collected within {LAB_ENCOUNTER_WINDOW.days} days of encounter start",
                            record_id=audit_id)
        else:
            cleaned = clean_text(rd.text)
            if rd.source_format == "txt" and cleaned != rd.text.strip():
                doc.log("text", "normalized", reason="OCR artifacts removed (hyphenation, soft wraps, spacing)",
                        record_id=audit_id)
            doc.text = cleaned

        # De-duplicate on patient + document family + day + normalized text.
        family = "lab" if rd.kind is DocKind.lab_report else "img" if rd.kind is DocKind.imaging_report else "note"
        day = doc.date.date().isoformat() if doc.date else ""
        doc.fingerprint = fingerprint(pid, family, day, doc.text)
        doc.doc_id = "doc-" + doc.fingerprint
        if (kept := seen.get(doc.fingerprint)) is not None:
            ds.dataset_audit.append(AuditEntry(
                record_id=audit_id, field="document", action="deduplicated", before=doc.title,
                after=kept.doc_id,
                reason=("same content as " + kept.sources[0] +
                        (" (cross-format duplicate)" if kept.source_format != doc.source_format else "")),
            ))
            kept.sources.append(audit_id)
            continue
        seen[doc.fingerprint] = doc

    ds.documents = sorted(seen.values(), key=lambda d: (d.patient_id, d.date or datetime.min))
    return ds


def write_outputs(ds: CleanDataset, out_dir: Path) -> dict[str, Path]:
    """Persist the clean dataset and a flat audit log for review."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "dataset": out_dir / "clean_dataset.json",
        "audit": out_dir / "audit_log.jsonl",
        "summary": out_dir / "ingestion_summary.json",
    }
    paths["dataset"].write_text(ds.model_dump_json(indent=2))
    with paths["audit"].open("w") as fh:
        for e in ds.all_audit():
            fh.write(e.model_dump_json() + "\n")
    paths["summary"].write_text(json.dumps({**ds.stats(), "issues_detail": [i.model_dump() for i in ds.issues]},
                                           indent=2, default=str))
    return paths
