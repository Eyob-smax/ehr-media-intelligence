"""Turn FHIR resources back into human-readable records.

Summaries and search both work from the FHIR Bundles (the system of record
after Task 2), not from the intermediate dataset, so whatever we show a
clinician is traceable to a stored FHIR resource.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Iterator

CLINICAL_TYPES = ("DocumentReference", "DiagnosticReport")


@dataclass(frozen=True)
class ResourceRecord:
    resource_type: str
    resource_id: str
    patient_ref: str
    date: str | None          # ISO date (YYYY-MM-DD)
    title: str
    category: str             # e.g. "Discharge summary", "LAB", "RAD"
    text: str
    conclusion: str | None = None
    encounter_ref: str | None = None


def _decode(att: dict[str, Any]) -> str:
    data = att.get("data")
    if not data:
        return ""
    try:
        return base64.b64decode(data).decode("utf-8")
    except Exception:  # noqa: BLE001 - malformed attachment should not break retrieval
        return ""


def _date(res: dict[str, Any]) -> str | None:
    for key in ("effectiveDateTime", "date", "issued"):
        if res.get(key):
            return str(res[key])[:10]
    period = (res.get("context") or {}).get("period") or {}
    return str(period["start"])[:10] if period.get("start") else None


def _coding_display(concept: dict[str, Any] | None) -> str:
    if not concept:
        return ""
    for c in concept.get("coding") or []:
        if c.get("display"):
            return c["display"]
        if c.get("code"):
            return c["code"]
    return concept.get("text", "")


def resource_record(res: dict[str, Any]) -> ResourceRecord | None:
    rt = res.get("resourceType")
    if rt == "DocumentReference":
        att = ((res.get("content") or [{}])[0]).get("attachment") or {}
        enc = ((res.get("context") or {}).get("encounter") or [{}])[0].get("reference")
        return ResourceRecord(
            resource_type=rt, resource_id=res["id"], patient_ref=(res.get("subject") or {}).get("reference", ""),
            date=_date(res), title=res.get("description") or (res.get("type") or {}).get("text", "Document"),
            category=_coding_display(res.get("type")) or "Clinical note", text=_decode(att), encounter_ref=enc,
        )
    if rt == "DiagnosticReport":
        att = (res.get("presentedForm") or [{}])[0]
        cat = ((res.get("category") or [{}])[0].get("coding") or [{}])[0].get("code", "")
        return ResourceRecord(
            resource_type=rt, resource_id=res["id"], patient_ref=(res.get("subject") or {}).get("reference", ""),
            date=_date(res), title=(res.get("code") or {}).get("text") or _coding_display(res.get("code")),
            category={"LAB": "Laboratory", "RAD": "Imaging"}.get(cat, cat or "Report"), text=_decode(att),
            conclusion=res.get("conclusion"), encounter_ref=(res.get("encounter") or {}).get("reference"),
        )
    return None


def bundle_patient(bundle: dict[str, Any]) -> dict[str, Any]:
    return next(e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Patient")


def patient_display(patient: dict[str, Any]) -> tuple[str, str]:
    """(name, MRN) for a Patient resource."""
    name = (patient.get("name") or [{}])[0]
    display = " ".join([*(name.get("given") or []), name.get("family", "")]).strip()
    mrn = next((i["value"] for i in patient.get("identifier", []) if i.get("use") == "official"), "")
    return display or mrn, mrn


def bundle_records(bundle: dict[str, Any]) -> Iterator[ResourceRecord]:
    for e in bundle.get("entry", []):
        rec = resource_record(e.get("resource") or {})
        if rec:
            yield rec


def encounters(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    return [e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Encounter"]


def observations(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    return [e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "Observation"]
