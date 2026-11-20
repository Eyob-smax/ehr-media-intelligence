"""Task 1: ingest heterogeneous raw EHR exports and clean them into a canonical, audited schema."""

from .models import (  # noqa: F401
    AuditEntry,
    CleanDataset,
    ClinicalDocument,
    DocKind,
    Encounter,
    Gender,
    LabObservation,
    Patient,
)
from .pipeline import run_ingestion  # noqa: F401
