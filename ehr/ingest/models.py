"""Canonical intermediate representation produced by the cleaning pipeline.

These models are deliberately FHIR-agnostic: they describe *what we know* about a
patient after cleaning, and the FHIR mapper (Task 2) decides how to express it.
Each record carries its own audit trail so a reviewer can see exactly what the
pipeline changed and why.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Gender(str, Enum):
    """FHIR administrative-gender value set."""

    male = "male"
    female = "female"
    other = "other"
    unknown = "unknown"


class DocKind(str, Enum):
    discharge_summary = "discharge_summary"
    progress_note = "progress_note"
    ed_note = "ed_note"
    imaging_report = "imaging_report"
    lab_report = "lab_report"
    scanned_document = "scanned_document"

    @property
    def is_diagnostic(self) -> bool:
        """Diagnostic results map to DiagnosticReport; narrative documents map to DocumentReference."""
        return self in (DocKind.imaging_report, DocKind.lab_report)


AuditAction = Literal[
    "normalized",   # value reformatted to canonical form
    "inferred",     # value derived from another field or source
    "filled",       # missing value populated from a fallback
    "missing",      # required value absent and left empty
    "dropped",      # value or record discarded as unusable
    "deduplicated", # record removed as a duplicate of another
    "merged",       # two identities combined into one patient
    "conflict",     # sources disagree; a winner was chosen
]


class AuditEntry(BaseModel):
    """One change the pipeline made to one record."""

    model_config = ConfigDict(frozen=True)

    record_id: str
    field: str
    action: AuditAction
    before: Any = None
    after: Any = None
    reason: str = ""
    source: str = ""


class Audited(BaseModel):
    audit: list[AuditEntry] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list, description="Raw files/records this was built from")

    def log(self, field: str, action: AuditAction, before: Any = None, after: Any = None,
            reason: str = "", source: str = "", record_id: str | None = None) -> None:
        self.audit.append(AuditEntry(
            record_id=record_id or self.record_key(), field=field, action=action,
            before=_jsonable(before), after=_jsonable(after), reason=reason, source=source,
        ))

    def record_key(self) -> str:  # overridden
        return "?"


def _jsonable(v: Any) -> Any:
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    if isinstance(v, Enum):
        return v.value
    return v


class Patient(Audited):
    patient_id: str = Field(description="Stable internal id derived from the canonical MRN")
    mrn: str = Field(pattern=r"^MRN-\d{7}$")
    alternate_mrns: list[str] = Field(default_factory=list, description="Other MRNs merged into this identity")
    given_name: str | None = None
    family_name: str | None = None
    birth_date: date | None = None
    gender: Gender = Gender.unknown

    def record_key(self) -> str:
        return f"Patient/{self.patient_id}"

    @property
    def display_name(self) -> str:
        parts = [p for p in (self.given_name, self.family_name) if p]
        return " ".join(parts) if parts else self.mrn


class Encounter(Audited):
    encounter_id: str
    patient_id: str
    start: datetime | None = None
    encounter_class: Literal["inpatient", "outpatient", "emergency", "unknown"] = "unknown"
    reason: str | None = None

    def record_key(self) -> str:
        return f"Encounter/{self.encounter_id}"


class LabObservation(BaseModel):
    test: str
    value: float | None = None
    unit: str | None = None
    ref_range: str | None = None
    flag: Literal["H", "L", "N"] | None = None
    loinc: str | None = None

    @property
    def is_abnormal(self) -> bool:
        return self.flag in ("H", "L")


class ClinicalDocument(Audited):
    doc_id: str
    patient_id: str
    encounter_id: str | None = None
    kind: DocKind
    title: str
    date: datetime | None = None
    text: str
    source_format: Literal["json", "csv", "txt"]
    observations: list[LabObservation] = Field(default_factory=list)
    fingerprint: str = Field(default="", description="Content hash used for de-duplication")

    def record_key(self) -> str:
        return f"Document/{self.doc_id}"


class DatasetIssue(BaseModel):
    severity: Literal["warning", "error"]
    record_id: str
    message: str


class CleanDataset(BaseModel):
    patients: list[Patient] = Field(default_factory=list)
    encounters: list[Encounter] = Field(default_factory=list)
    documents: list[ClinicalDocument] = Field(default_factory=list)
    dataset_audit: list[AuditEntry] = Field(default_factory=list, description="Record-level events (dedup, merges, drops)")
    issues: list[DatasetIssue] = Field(default_factory=list)

    def all_audit(self) -> list[AuditEntry]:
        out = list(self.dataset_audit)
        for group in (self.patients, self.encounters, self.documents):
            for rec in group:
                out.extend(rec.audit)
        return out

    def stats(self) -> dict[str, Any]:
        actions: dict[str, int] = {}
        for e in self.all_audit():
            actions[e.action] = actions.get(e.action, 0) + 1
        return {
            "patients": len(self.patients),
            "encounters": len(self.encounters),
            "documents": len(self.documents),
            "documents_by_kind": _count(d.kind.value for d in self.documents),
            "audit_actions": dict(sorted(actions.items())),
            "issues": len(self.issues),
        }


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return dict(sorted(out.items()))
