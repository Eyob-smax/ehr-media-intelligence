"""Tests for FHIR mapping, Bundle validation and storage (Task 2)."""

import copy

from ehr.fhir.mapping import build_bundles
from ehr.fhir.store import BundleStore, record_hash
from ehr.fhir.text import bundle_records
from ehr.fhir.validation import build_report, report_markdown, validate_bundle


def _by_type(bundle, rt):
    return [e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == rt]


def test_every_sample_bundle_is_valid(sample_bundles):
    results = [validate_bundle(b, pid) for pid, b in sample_bundles.items()]
    errors = [i for r in results for i in r.issues if i.severity == "error"]
    assert not errors, errors[:3]
    assert len(results) == 16


def test_bundle_contains_required_resource_types(sample_bundles):
    types = {e["resource"]["resourceType"] for b in sample_bundles.values() for e in b["entry"]}
    assert {"Patient", "DocumentReference", "DiagnosticReport", "Encounter", "Observation"} <= types


def test_subject_and_encounter_references_resolve_inside_bundle(sample_bundles):
    for b in sample_bundles.values():
        urls = {e["fullUrl"] for e in b["entry"]}
        patient_url = next(e["fullUrl"] for e in b["entry"] if e["resource"]["resourceType"] == "Patient")
        for dr in _by_type(b, "DiagnosticReport"):
            assert dr["subject"]["reference"] == patient_url
            if "encounter" in dr:
                assert dr["encounter"]["reference"] in urls
            for r in dr.get("result", []):
                assert r["reference"] in urls
        for doc in _by_type(b, "DocumentReference"):
            assert doc["subject"]["reference"] == patient_url
            for enc in doc.get("context", {}).get("encounter", []):
                assert enc["reference"] in urls


def test_lab_reports_carry_coded_observations_with_interpretation(sample_bundles):
    labs = [dr for b in sample_bundles.values() for dr in _by_type(b, "DiagnosticReport")
            if dr["category"][0]["coding"][0]["code"] == "LAB"]
    assert labs and all(dr.get("result") for dr in labs)
    high = [o for b in sample_bundles.values() for o in _by_type(b, "Observation")
            if o.get("interpretation", [{}])[0].get("coding", [{}])[0].get("code") == "H"]
    assert high, "abnormal lab values must be flagged on the Observation"
    assert all(o["code"]["coding"][0]["system"] == "http://loinc.org" for o in high)


def test_merged_mrn_is_kept_as_old_identifier(sample_bundles):
    ids = [p["identifier"] for b in sample_bundles.values() for p in _by_type(b, "Patient")]
    assert any(any(i["use"] == "old" for i in idents) for idents in ids)


def test_bundles_are_deterministic(sample_dataset):
    a, b = build_bundles(sample_dataset), build_bundles(sample_dataset)
    assert {k: record_hash(v) for k, v in a.items()} == {k: record_hash(v) for k, v in b.items()}


def test_validator_reports_structure_terminology_and_reference_errors(sample_bundles):
    bundle = copy.deepcopy(next(iter(sample_bundles.values())))
    dr = next(e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "DiagnosticReport")
    dr["status"] = "bogus"                               # terminology (fhir.resources accepts this)
    dr["subject"] = {"reference": "urn:uuid:missing"}    # dangling reference
    doc = next(e["resource"] for e in bundle["entry"] if e["resource"]["resourceType"] == "DocumentReference")
    del doc["content"]                                   # structure: content is 1..*

    result = validate_bundle(bundle, "p1")
    layers = {i.layer for i in result.issues if i.severity == "error"}
    assert not result.valid
    assert layers == {"structure", "terminology", "reference"}

    md = report_markdown(build_report([result]))
    assert "bogus" in md and "dangling reference" in md


def test_store_roundtrip_and_resource_index(tmp_path, sample_bundles):
    store = BundleStore(tmp_path / "ehr.sqlite3")
    pid, bundle = next(iter(sample_bundles.items()))
    h = store.save(pid, bundle)
    assert store.get_bundle(pid) == bundle
    assert store.get_meta(pid)["record_hash"] == h
    resources = store.resources_for(pid)
    assert len(resources) == len(list(bundle_records(bundle)))
    assert store.get_resource(resources[0]["resource_id"])["id"] == resources[0]["resource_id"]
    store.save(pid, bundle)  # upsert does not duplicate rows
    assert store.count() == 1 and len(store.resources_for(pid)) == len(resources)
