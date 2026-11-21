"""Build the compact, citable patient context the summarizers read.

Records are labeled ``R1..Rn`` (newest first) and encounters ``E1..En`` so the
model can cite them, and so the grounding check can look up exactly which
text a claim should be supported by.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ..fhir.text import ResourceRecord, bundle_patient, bundle_records, encounters, patient_display


@dataclass
class SourceItem:
    sid: str
    kind: str          # imaging | lab | document | encounter
    date: str
    title: str
    text: str
    resource_type: str
    resource_id: str
    conclusion: str | None = None


@dataclass
class PatientContext:
    patient_id: str
    name: str
    mrn: str
    gender: str
    age: int | None
    items: list[SourceItem] = field(default_factory=list)

    def by_id(self) -> dict[str, SourceItem]:
        return {i.sid: i for i in self.items}

    def render(self) -> str:
        head = (f"Patient: {self.name} | MRN {self.mrn} | {self.gender}"
                + (f" | age {self.age}" if self.age is not None else ""))
        blocks = [head, ""]
        for i in self.items:
            if i.kind == "encounter":
                blocks.append(f"[{i.sid}] {i.date} ENCOUNTER {i.title}")
            else:
                blocks.append(f"[{i.sid}] {i.date} {i.kind.upper()} — {i.title}\n{i.text.strip()}")
            blocks.append("")
        return "\n".join(blocks).strip()


def _kind(r: ResourceRecord) -> str:
    if r.resource_type == "DiagnosticReport":
        return "lab" if r.category == "Laboratory" else "imaging"
    return "document"


def build_context(patient_id: str, bundle: dict[str, Any]) -> PatientContext:
    patient = bundle_patient(bundle)
    name, mrn = patient_display(patient)
    records = sorted(bundle_records(bundle), key=lambda r: (r.date or "", r.resource_id), reverse=True)

    latest = records[0].date if records and records[0].date else None
    age = None
    if patient.get("birthDate") and latest:
        b, ref = date.fromisoformat(patient["birthDate"]), date.fromisoformat(latest)
        age = ref.year - b.year - ((ref.month, ref.day) < (b.month, b.day))

    ctx = PatientContext(patient_id=patient_id, name=name, mrn=mrn, gender=patient.get("gender", "unknown"), age=age)
    for n, r in enumerate(records, start=1):
        ctx.items.append(SourceItem(
            sid=f"R{n}", kind=_kind(r), date=r.date or "unknown", title=r.title, text=r.text,
            resource_type=r.resource_type, resource_id=r.resource_id, conclusion=r.conclusion,
        ))
    encs = sorted(encounters(bundle), key=lambda e: (e.get("period") or {}).get("start", ""), reverse=True)
    for n, e in enumerate(encs, start=1):
        reason = ((e.get("reasonCode") or [{}])[0]).get("text", "unspecified reason")
        cls = (e.get("class") or {}).get("display", "")
        ctx.items.append(SourceItem(
            sid=f"E{n}", kind="encounter", date=str((e.get("period") or {}).get("start", "unknown"))[:10],
            title=f"{cls}: {reason}", text=reason, resource_type="Encounter", resource_id=e["id"],
        ))
    return ctx
