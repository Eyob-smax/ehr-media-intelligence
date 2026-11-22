import importlib.util
import os
import sys
from datetime import date
from pathlib import Path

import pytest

# Tests must never call a paid API or download models implicitly.
os.environ.setdefault("EHR_SUMMARIZER", "offline")
os.environ.setdefault("EHR_EMBEDDER", "hash")

ROOT = Path(__file__).resolve().parent.parent
REF = date(2025, 10, 1)


def _generator():
    spec = importlib.util.spec_from_file_location("generate_data", ROOT / "scripts" / "generate_data.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look the module up while the class body executes
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def curated_raw(tmp_path_factory) -> Path:
    """The 16 hand-built edge-case patients, generated fresh (deterministic) for each test session."""
    out = tmp_path_factory.mktemp("raw_curated")
    _generator().build(out, seed=7, cohort=0)
    return out


@pytest.fixture(scope="session")
def cohort_raw(tmp_path_factory) -> Path:
    """Curated patients plus a 60-patient random cohort with every source layout."""
    out = tmp_path_factory.mktemp("raw_cohort")
    _generator().build(out, seed=7, cohort=60)
    return out


@pytest.fixture(scope="session")
def sample_dataset(curated_raw):
    from ehr.ingest.pipeline import run_ingestion

    return run_ingestion(curated_raw, reference=REF)


@pytest.fixture(scope="session")
def sample_bundles(sample_dataset):
    from ehr.fhir.mapping import build_bundles

    return build_bundles(sample_dataset)
