import os
from datetime import date
from pathlib import Path

import pytest

# Tests must never call a paid API or download models implicitly.
os.environ.setdefault("EHR_SUMMARIZER", "offline")
os.environ.setdefault("EHR_EMBEDDER", "hash")

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
REF = date(2025, 10, 1)


@pytest.fixture(scope="session")
def sample_dataset():
    from ehr.ingest.pipeline import run_ingestion

    if not (RAW / "ehr_export.json").exists():
        pytest.skip("sample data not generated (run scripts/generate_data.py)")
    return run_ingestion(RAW, reference=REF)


@pytest.fixture(scope="session")
def sample_bundles(sample_dataset):
    from ehr.fhir.mapping import build_bundles

    return build_bundles(sample_dataset)
