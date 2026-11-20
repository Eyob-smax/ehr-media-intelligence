"""Integration tests for the cleaning pipeline (Task 1).

Each test writes a tiny raw dataset to a temp dir so the edge case under test
is obvious and independent of the generated sample data.
"""

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from ehr.ingest.models import DocKind, Gender
from ehr.ingest.pipeline import run_ingestion, write_outputs

REF = date(2025, 10, 1)
NOTE = "PROGRESS NOTE\nReason for Visit: Cough.\nAssessment:\n1. Viral bronchitis\nPlan: Supportive care."


def _record(rid, patient, enc_id="ENC-1", enc_date="2025-03-01", docs=None):
    return {
        "record_id": rid,
        "patient": patient,
        "encounter": {"id": enc_id, "date": enc_date, "class": "outpatient", "reason": "Cough"},
        "documents": docs if docs is not None else [
            {"doc_type": "progress_note", "title": "Office Visit", "date": enc_date, "text": NOTE}],
    }


def _write(tmp_path: Path, records=None, csv_rows=None, notes=None) -> Path:
    raw = tmp_path / "raw"
    (raw / "notes").mkdir(parents=True)
    if records is not None:
        (raw / "export.json").write_text(json.dumps({"records": records}))
    if csv_rows is not None:
        header = "mrn,patient_name,dob,collected,panel,test,value,unit,ref_range,flag"
        (raw / "labs.csv").write_text("\n".join([header, *csv_rows]) + "\n")
    for name, body in (notes or {}).items():
        (raw / "notes" / name).write_text(body)
    return raw


JANE = {"mrn": "MRN-0000042", "name": "DOE, JANE", "dob": "1970-05-06", "gender": "F"}


def test_exact_and_whitespace_duplicates_are_removed_and_logged(tmp_path):
    spaced = NOTE.replace(". ", ".   ").upper()
    raw = _write(tmp_path, records=[
        _record("R1", JANE),
        _record("R1", JANE),  # re-sent export message
        _record("R2", JANE, enc_id="ENC-2", docs=[
            {"doc_type": "progress_note", "title": "Office Visit", "date": "03/01/2025", "text": f"  {spaced}\n"}]),
    ])
    ds = run_ingestion(raw, reference=REF)

    assert len(ds.patients) == 1
    assert len(ds.documents) == 1, "all three notes are the same content"
    dedup = [e for e in ds.dataset_audit if e.action == "deduplicated" and e.field == "document"]
    assert len(dedup) == 2
    assert set(ds.documents[0].sources) == {"json:R1/doc0", "json:R2/doc0"}


def test_cross_format_duplicate_scanned_copy_is_removed(tmp_path):
    scan = ("PATIENT: DOE, JANE    MRN: 42\nDOB: 5/6/70    SEX: F\nDATE OF SERVICE: March 1, 2025\n"
            "DOCUMENT TYPE: Progress Note\n" + "-" * 40 + "\n" +
            "PROGRESS NOTE\nReason for Visit: Cough.\nAssess-\nment:\n1. Viral bronchitis\nPlan: Supportive   care.\n\f")
    ds = run_ingestion(_write(tmp_path, records=[_record("R1", JANE)], notes={"scan.txt": scan}), reference=REF)

    assert len(ds.documents) == 1
    kept = ds.documents[0]
    assert kept.source_format == "json", "the EHR export copy (with encounter link) wins"
    assert any("cross-format" in e.reason for e in ds.dataset_audit)


def test_conflicting_mrn_typo_is_merged_by_name_and_birth_date(tmp_path):
    raw = _write(
        tmp_path,
        records=[_record("R1", JANE)],
        csv_rows=[
            # lab system has the last two digits transposed: 0000024 instead of 0000042
            "0000024,\"Doe, Jane\",05/06/1970,03/01/2025 10:00,CBC,Hemoglobin,13.1,g/dL,12.0-17.5,",
        ],
    )
    ds = run_ingestion(raw, reference=REF)

    assert len(ds.patients) == 1
    p = ds.patients[0]
    assert p.mrn == "MRN-0000042", "the EHR export MRN is authoritative"
    assert p.alternate_mrns == ["MRN-0000024"]
    assert {d.patient_id for d in ds.documents} == {p.patient_id}
    merged = [e for e in ds.dataset_audit if e.action == "merged"]
    assert merged and merged[0].before == "MRN-0000024"


def test_same_mrn_different_people_are_not_merged_by_name_alone(tmp_path):
    other = {"mrn": "77", "name": "Jane Doe", "dob": "1988-01-01", "gender": "F"}
    ds = run_ingestion(_write(tmp_path, records=[_record("R1", JANE), _record("R2", other, enc_id="ENC-9")]),
                       reference=REF)
    assert len(ds.patients) == 2, "same name but different DOB and MRN must stay separate"


def test_birth_date_conflict_keeps_majority_and_flags_day_month_swap(tmp_path):
    swapped = {**JANE, "dob": "1970-06-05"}
    raw = _write(tmp_path, records=[
        _record("R1", JANE), _record("R2", JANE, enc_id="ENC-2", enc_date="2025-04-01"),
        _record("R3", swapped, enc_id="ENC-3", enc_date="2025-05-01"),
    ])
    p = run_ingestion(raw, reference=REF).patients[0]

    assert p.birth_date == date(1970, 5, 6)
    conflict = next(a for a in p.audit if a.field == "birth_date" and a.action == "conflict")
    assert "transposition" in conflict.reason
    assert conflict.before == {"1970-05-06": 2, "1970-06-05": 1}


def test_missing_fields_are_filled_or_flagged(tmp_path):
    no_gender = {"mrn": "42", "first_name": "jane", "last_name": "DOE", "DOB": "May 6, 1970"}
    raw = _write(tmp_path, records=[_record("R1", no_gender, docs=[
        {"doc_type": "ed_note", "title": "ED Note", "text": NOTE},  # no date at all
    ])], csv_rows=[
        "42,,1970-05-06,2025-03-01 09:00,BMP,Sodium,,mmol/L,135-145,",       # missing name and value
        "42,DOE JANE,1970-05-06,2025-03-01 09:00,BMP,Sodium,131,mmol/L,135-145,",  # later row supplies value
    ])
    ds = run_ingestion(raw, reference=REF)
    p = ds.patients[0]

    assert p.display_name == "Jane Doe"
    assert p.gender is Gender.unknown
    assert any(a.field == "gender" and a.action == "missing" for a in p.audit)

    note = next(d for d in ds.documents if d.kind is DocKind.ed_note)
    assert note.date == datetime(2025, 3, 1)
    assert any(a.field == "date" and a.action == "filled" for a in note.audit)

    lab = next(d for d in ds.documents if d.kind is DocKind.lab_report)
    sodium = lab.observations[0]
    assert sodium.value == 131 and sodium.flag == "L", "empty value replaced and flag derived from range"
    assert lab.encounter_id == "ENC-1", "lab linked to the same-day encounter"


def test_unattributable_records_are_reported_not_silently_dropped(tmp_path):
    ds = run_ingestion(_write(tmp_path, records=[_record("R1", {"mrn": "N/A", "name": "Ghost"})]), reference=REF)
    assert ds.patients == []
    assert any(i.severity == "error" for i in ds.issues)


@pytest.mark.skipif(not (Path(__file__).parent.parent / "data/raw/ehr_export.json").exists(),
                    reason="sample data not generated")
def test_sample_dataset_end_to_end(tmp_path):
    raw = Path(__file__).parent.parent / "data/raw"
    ds = run_ingestion(raw, reference=REF)
    assert len(ds.patients) == 16
    assert len(ds.documents) >= 50, "assessment asks for search over at least 50 records"
    assert all(d.date for d in ds.documents)
    paths = write_outputs(ds, tmp_path)
    assert sum(1 for _ in paths["audit"].open()) == len(ds.all_audit())
