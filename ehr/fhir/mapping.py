"""Canonical models -> FHIR R4 resource dictionaries, grouped into one Bundle per patient.

Mapping choices
- Patient       <- Patient (MRN as an ``MR``-typed identifier; merged MRNs kept with use=old)
- Encounter     <- Encounter (needed so subject *and* encounter references resolve)
- DocumentReference <- narrative documents (discharge summaries, notes, scanned records)
- DiagnosticReport  <- lab reports (category LAB, with Observation results) and
                       imaging reports (category RAD, impression as ``conclusion``)
- Observation   <- each lab analyte, so abnormal values are machine-readable

All resources get deterministic UUIDv5 ids derived from our internal ids, so
re-running the pipeline produces identical Bundles (stable cache keys, clean diffs).
Inside a Bundle every reference is a ``urn:uuid:`` that matches an entry fullUrl.
"""

from __future__ import annotations

import base64
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from ..ingest.labs import find_panel
from ..ingest.models import CleanDataset, ClinicalDocument, DocKind, Encounter, LabObservation, Patient

NS = uuid.UUID("6f1c1d1e-5b9a-4c55-9a51-0e8f3b8b7a10")
MRN_SYSTEM = "https://ehr.example.org/fhir/sid/mrn"

LOINC = "http://loinc.org"
UCUM = "http://unitsofmeasure.org"
V2_0203 = "http://terminology.hl7.org/CodeSystem/v2-0203"
V2_0074 = "http://terminology.hl7.org/CodeSystem/v2-0074"
V3_ACT = "http://terminology.hl7.org/CodeSystem/v3-ActCode"
V3_INTERP = "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation"
OBS_CAT = "http://terminology.hl7.org/CodeSystem/observation-category"
DATA_ABSENT = "http://terminology.hl7.org/CodeSystem/data-absent-reason"
US_CORE_DOC_CAT = "http://hl7.org/fhir/us/core/CodeSystem/us-core-documentreference-category"

DOC_TYPE_LOINC: dict[DocKind, tuple[str, str] | None] = {
    DocKind.discharge_summary: ("18842-5", "Discharge summary"),
    DocKind.progress_note: ("11506-3", "Progress note"),
    DocKind.ed_note: ("34111-5", "Emergency department Note"),
    DocKind.scanned_document: None,  # free-text outside records: no reliable single LOINC type
}

IMAGING_LOINC = ("18748-4", "Diagnostic imaging study")

ENCOUNTER_CLASS = {
    "inpatient": ("IMP", "inpatient encounter"),
    "outpatient": ("AMB", "ambulatory"),
    "emergency": ("EMER", "emergency"),
    "unknown": ("AMB", "ambulatory"),
}

UCUM_FIX = {"mIU/L": "m[IU]/L"}


def rid(kind: str, key: str) -> str:
    return str(uuid.uuid5(NS, f"{kind}/{key}"))


def ref(resource: dict[str, Any]) -> dict[str, str]:
    return {"reference": f"urn:uuid:{resource['id']}"}


def _instant(dt: datetime | None) -> str | None:
    """FHIR ``instant`` requires a timezone. Source timestamps are naive; we treat them as UTC."""
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _datetime(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.hour == dt.minute == dt.second == 0:
        return dt.date().isoformat()
    return _instant(dt)


def _attachment(text: str, title: str, created: datetime | None) -> dict[str, Any]:
    att = {
        "contentType": "text/plain; charset=utf-8",
        "language": "en-US",
        "data": base64.b64encode(text.encode("utf-8")).decode("ascii"),
        "title": title,
    }
    if created:
        att["creation"] = _datetime(created)
    return att


def _section(text: str, name: str) -> str | None:
    m = re.search(rf"^{name}:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Resource builders
# ---------------------------------------------------------------------------


def patient_resource(p: Patient) -> dict[str, Any]:
    identifiers = [{
        "use": "official",
        "type": {"coding": [{"system": V2_0203, "code": "MR", "display": "Medical record number"}]},
        "system": MRN_SYSTEM,
        "value": p.mrn,
    }]
    identifiers += [{"use": "old", "system": MRN_SYSTEM, "value": alt} for alt in p.alternate_mrns]
    res: dict[str, Any] = {
        "resourceType": "Patient",
        "id": rid("Patient", p.patient_id),
        "identifier": identifiers,
        "gender": p.gender.value,
    }
    if p.family_name or p.given_name:
        name: dict[str, Any] = {"use": "official"}
        if p.family_name:
            name["family"] = p.family_name
        if p.given_name:
            name["given"] = [p.given_name]
        res["name"] = [name]
    if p.birth_date:
        res["birthDate"] = p.birth_date.isoformat()
    return res


def encounter_resource(e: Encounter, patient: dict[str, Any]) -> dict[str, Any]:
    code, display = ENCOUNTER_CLASS[e.encounter_class]
    res: dict[str, Any] = {
        "resourceType": "Encounter",
        "id": rid("Encounter", e.encounter_id),
        "identifier": [{"system": "https://ehr.example.org/fhir/sid/encounter", "value": e.encounter_id}],
        "status": "finished",
        "class": {"system": V3_ACT, "code": code, "display": display},
        "subject": ref(patient),
    }
    if e.start:
        res["period"] = {"start": _datetime(e.start)}
    if e.reason:
        res["reasonCode"] = [{"text": e.reason}]
    return res


def document_reference(d: ClinicalDocument, patient: dict, encounter: dict | None) -> dict[str, Any]:
    loinc = DOC_TYPE_LOINC.get(d.kind)
    doc_type: dict[str, Any] = {"text": d.title}
    if loinc:
        doc_type["coding"] = [{"system": LOINC, "code": loinc[0], "display": loinc[1]}]
    res: dict[str, Any] = {
        "resourceType": "DocumentReference",
        "id": rid("DocumentReference", d.doc_id),
        "identifier": [{"system": "urn:ietf:rfc:3986", "value": f"urn:uuid:{rid('DocumentReference', d.doc_id)}"}],
        "status": "current",
        "docStatus": "final",
        "type": doc_type,
        "category": [{"coding": [{"system": US_CORE_DOC_CAT, "code": "clinical-note", "display": "Clinical Note"}]}],
        "subject": ref(patient),
        "description": d.title,
        "content": [{"attachment": _attachment(d.text, d.title, d.date)}],
    }
    if d.date:
        res["date"] = _instant(d.date)
    if encounter is not None:
        res["context"] = {"encounter": [ref(encounter)]}
        if d.date:
            res["context"]["period"] = {"start": _datetime(d.date)}
    return res


def observation_resource(d: ClinicalDocument, o: LabObservation, idx: int, patient: dict,
                         encounter: dict | None) -> dict[str, Any]:
    code: dict[str, Any] = {"text": o.test}
    if o.loinc:
        code["coding"] = [{"system": LOINC, "code": o.loinc, "display": o.test}]
    res: dict[str, Any] = {
        "resourceType": "Observation",
        "id": rid("Observation", f"{d.doc_id}/{idx}"),
        "status": "final",
        "category": [{"coding": [{"system": OBS_CAT, "code": "laboratory", "display": "Laboratory"}]}],
        "code": code,
        "subject": ref(patient),
    }
    if encounter is not None:
        res["encounter"] = ref(encounter)
    if d.date:
        res["effectiveDateTime"] = _datetime(d.date)
    if o.value is not None:
        q: dict[str, Any] = {"value": o.value}
        if o.comparator:
            q["comparator"] = o.comparator
        if o.unit:
            q.update(unit=o.unit, system=UCUM, code=UCUM_FIX.get(o.unit, o.unit))
        res["valueQuantity"] = q
    else:
        res["dataAbsentReason"] = {"coding": [{"system": DATA_ABSENT, "code": "unknown", "display": "Unknown"}]}
    if o.flag:
        disp = {"H": "High", "L": "Low", "N": "Normal", "HH": "Critical high", "LL": "Critical low",
                "A": "Abnormal"}[o.flag]
        res["interpretation"] = [{"coding": [{"system": V3_INTERP, "code": o.flag, "display": disp}]}]
    if o.ref_range:
        res["referenceRange"] = [{"text": o.ref_range}]
    return res


def _abnormal_phrase(o: LabObservation) -> str:
    direction = {"H": "high", "L": "low", "HH": "critical high", "LL": "critical low"}.get(o.flag or "", "abnormal")
    if o.value is None:  # flagged by the source but the value itself was not reported
        return f"{o.test} flagged {direction} (value not reported)"
    return f"{o.test} {o.comparator or ''}{o.value:g} {o.unit or ''}".rstrip() + f" ({direction})"


def diagnostic_report(d: ClinicalDocument, patient: dict, encounter: dict | None,
                      results: list[dict]) -> dict[str, Any]:
    is_lab = d.kind is DocKind.lab_report
    if is_lab:
        panel = find_panel(d.title.removesuffix(" Results"))
        code_tuple = (panel.loinc, panel.loinc_display) if panel else None
        category = {"system": V2_0074, "code": "LAB", "display": "Laboratory"}
        abnormal = [o for o in d.observations if o.is_abnormal]
        conclusion = ("Abnormal: " + "; ".join(_abnormal_phrase(o) for o in abnormal)
                      if abnormal else "All reported results within reference range.")
    else:
        code_tuple = IMAGING_LOINC
        category = {"system": V2_0074, "code": "RAD", "display": "Radiology"}
        conclusion = _section(d.text, "Impression")
    code: dict[str, Any] = {"text": d.title}
    if code_tuple:
        code["coding"] = [{"system": LOINC, "code": code_tuple[0], "display": code_tuple[1]}]
    res: dict[str, Any] = {
        "resourceType": "DiagnosticReport",
        "id": rid("DiagnosticReport", d.doc_id),
        "status": "final",
        "category": [{"coding": [category]}],
        "code": code,
        "subject": ref(patient),
        "presentedForm": [_attachment(d.text, d.title, d.date)],
    }
    if encounter is not None:
        res["encounter"] = ref(encounter)
    if d.date:
        res["effectiveDateTime"] = _datetime(d.date)
        res["issued"] = _instant(d.date)
    if results:
        res["result"] = [ref(r) for r in results]
    if conclusion:
        res["conclusion"] = conclusion
    return res


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------


def _entry(resource: dict[str, Any]) -> dict[str, Any]:
    return {"fullUrl": f"urn:uuid:{resource['id']}", "resource": resource}


def build_patient_bundle(ds: CleanDataset, patient: Patient) -> dict[str, Any]:
    """One ``collection`` Bundle containing the patient and everything that references them."""
    pat = patient_resource(patient)
    entries = [_entry(pat)]
    enc_map: dict[str, dict] = {}
    for e in sorted((e for e in ds.encounters if e.patient_id == patient.patient_id),
                    key=lambda e: e.start or datetime.min):
        enc_res = encounter_resource(e, pat)
        enc_map[e.encounter_id] = enc_res
        entries.append(_entry(enc_res))

    for d in (d for d in ds.documents if d.patient_id == patient.patient_id):
        enc_res = enc_map.get(d.encounter_id) if d.encounter_id else None
        if d.kind.is_diagnostic:
            obs = [observation_resource(d, o, i, pat, enc_res) for i, o in enumerate(d.observations)]
            entries.append(_entry(diagnostic_report(d, pat, enc_res, obs)))
            entries.extend(_entry(o) for o in obs)
        else:
            entries.append(_entry(document_reference(d, pat, enc_res)))

    return {
        "resourceType": "Bundle",
        "id": rid("Bundle", patient.patient_id),
        "identifier": {"system": MRN_SYSTEM, "value": patient.mrn},
        "type": "collection",
        "timestamp": _instant(max((d.date for d in ds.documents if d.patient_id == patient.patient_id and d.date),
                                  default=datetime(1970, 1, 1))),
        "entry": entries,
    }


def build_bundles(ds: CleanDataset) -> dict[str, dict[str, Any]]:
    """patient_id -> Bundle dict."""
    return {p.patient_id: build_patient_bundle(ds, p) for p in ds.patients}
