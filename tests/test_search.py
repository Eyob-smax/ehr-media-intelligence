"""Tests for semantic search and the API (Task 4).

Default runs use the deterministic hash embedder (no model download). The
``slow`` test exercises the real all-MiniLM-L6-v2 model for semantic quality
and the <2 s latency requirement: ``pytest -m slow``.
"""

import time
from datetime import date

import pytest
from fastapi.testclient import TestClient

from ehr.api import create_app
from ehr.config import Settings
from ehr.fhir.store import BundleStore
from ehr.search.embeddings import HashEmbedder, SentenceTransformerEmbedder
from ehr.search.index import SearchIndex, chunk_words
from ehr.search.synonyms import expand
from ehr.summarize.cache import SummaryCache
from ehr.summarize.offline import OfflineSummarizer
from ehr.summarize.service import SummaryService


@pytest.fixture(scope="module")
def built(tmp_path_factory, sample_bundles):
    """A populated store, summaries and settings pointing at temp locations."""
    root = tmp_path_factory.mktemp("search")
    settings = Settings(db_path=root / "ehr.sqlite3", chroma_dir=root / "chroma", out_dir=root / "out",
                        summarizer="offline", embedder="hash")
    store = BundleStore(settings.db_path)
    for pid, b in sample_bundles.items():
        store.save(pid, b)
    summaries = {s.patient_id: s for s in
                 SummaryService(store, SummaryCache(settings.db_path), OfflineSummarizer()).summarize_all()}
    return settings, store, summaries


@pytest.fixture(scope="module")
def index(built):
    settings, store, summaries = built
    idx = SearchIndex(settings.chroma_dir, HashEmbedder())
    idx.rebuild(store.iter_bundles(), summaries)
    return idx


def _name(sample_dataset, family):
    return next(p.display_name for p in sample_dataset.patients if p.family_name == family)


# --- index ---------------------------------------------------------------------


def test_chunking_overlaps_and_covers_text():
    words = [f"w{i}" for i in range(300)]
    chunks = chunk_words(" ".join(words), size=120, overlap=30)
    assert chunks[0].split()[0] == "w0" and chunks[-1].split()[-1] == "w299"
    assert chunks[1].split()[0] == "w90"


def test_synonym_expansion_maps_lay_terms_to_chart_vocabulary():
    text, groups = expand("blood clot in the leg")
    assert "thrombus" in text
    assert ["blood clot", "thrombus", "thrombosis", "dvt", "embolism"] in groups
    assert ["leg"] in groups and ["the"] not in groups


def test_relevant_record_ranks_first(index, sample_dataset):
    hits = index.search("right lower lobe pneumonia chest x-ray").hits
    assert hits[0].patient_name == _name(sample_dataset, "Banks")
    assert len(hits) == 5
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)
    assert all(0 <= h.score <= 1 for h in hits)


def test_one_hit_per_resource(index):
    hits = index.search("discharge summary hospital course", top_k=10).hits
    ids = [h.resource_id for h in hits]
    assert len(ids) == len(set(ids))


def test_resource_type_filter(index):
    hits = index.search("heart failure", resource_types=["DiagnosticReport"], top_k=10).hits
    assert hits and {h.resource_type for h in hits} == {"DiagnosticReport"}
    hits = index.search("heart failure", resource_types=["Patient"]).hits
    assert hits and {h.title for h in hits} == {"AI clinical summary"}


def test_date_range_filter(index):
    hits = index.search("lab results", date_from=date(2025, 8, 1), date_to=date(2025, 8, 31), top_k=20).hits
    assert hits and all("2025-08-01" <= h.date <= "2025-08-31" for h in hits)


def test_mrn_query_filters_to_that_patient(index, sample_dataset):
    p = next(p for p in sample_dataset.patients if p.family_name == "Quintero")
    number = p.mrn.removeprefix("MRN-").lstrip("0")
    hits = index.search(f"MRN {number}", top_k=10).hits
    assert hits and {h.patient_id for h in hits} == {p.patient_id}


def test_search_latency_under_two_seconds_for_50_plus_records(index, built):
    _, store, _ = built
    n_records = sum(len(store.resources_for(p["patient_id"])) for p in store.list_patients())
    assert n_records >= 50
    queries = ["shortness of breath", "abnormal kidney function", "MRI findings", "elevated white count",
               "anticoagulation", "thyroid"]
    t0 = time.perf_counter()
    for q in queries:
        index.search(q)
    assert (time.perf_counter() - t0) / len(queries) < 2.0


# --- API -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client(built, index):
    settings, _, _ = built
    app = create_app(settings, embedder=HashEmbedder(), frontend_dir=None)
    with TestClient(app) as c:
        yield c


def test_search_endpoint_contract(client):
    r = client.post("/search", json={"query": "kidney stone"})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == len(body["results"]) <= 5
    first = body["results"][0]
    for key in ("rank", "score", "patient_name", "mrn", "resource_type", "date", "snippet", "summary_snippet"):
        assert key in first
    assert first["rank"] == 1 and first["mrn"].startswith("MRN-")


def test_search_filters_are_query_parameters(client):
    r = client.post("/search?resource_type=DiagnosticReport&date_from=2025-06-01&date_to=2025-12-31",
                    json={"query": "imaging impression", "top_k": 10})
    assert r.status_code == 200
    res = r.json()["results"]
    assert res and all(x["resource_type"] == "DiagnosticReport" and x["date"] >= "2025-06-01" for x in res)
    assert r.json()["filters"]["resource_types"] == ["DiagnosticReport"]


@pytest.mark.parametrize("url, body", [
    ("/search", {"query": ""}),
    ("/search", {"query": "x", "top_k": 0}),
    ("/search?resource_type=Banana", {"query": "x"}),
    ("/search?date_from=2025-09-01&date_to=2025-01-01", {"query": "x"}),
])
def test_search_rejects_bad_input(client, url, body):
    assert client.post(url, json=body).status_code == 422


def test_patient_detail_has_summary_disclaimer_and_linked_resources(client, sample_dataset):
    pid = next(p.patient_id for p in sample_dataset.patients if p.family_name == "Lindqvist")
    r = client.get(f"/patients/{pid}")
    assert r.status_code == 200
    d = r.json()
    assert d["summary"]["disclaimer"] and d["summary"]["chief_concern"]
    assert d["alternate_mrns"], "merged MRN is exposed"
    assert len(d["resources"]) >= 3
    rid = d["resources"][0]["resource_id"]
    assert client.get(f"/resources/{rid}").json()["id"] == rid
    assert client.get(f"/patients/{pid}/bundle").json()["resourceType"] == "Bundle"


def test_unknown_ids_return_404(client):
    assert client.get("/patients/nope").status_code == 404
    assert client.get("/patients/nope/bundle").status_code == 404
    assert client.get("/resources/nope").status_code == 404


# --- real model (opt-in) ----------------------------------------------------------


@pytest.mark.slow
def test_real_model_semantic_quality_and_latency(built, sample_dataset, tmp_path):
    _, store, summaries = built
    idx = SearchIndex(tmp_path / "chroma", SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2"))
    idx.rebuild(store.iter_bundles(), summaries)
    cases = {
        "blood clot in the leg": "Watanabe",
        "kidney stone": "Castellanos",
        "fluid on the lungs and enlarged heart": "Quintero",
        "lung infection": "Banks",
    }
    worst = 0.0
    for q, family in cases.items():
        t0 = time.perf_counter()
        hits = idx.search(q).hits
        worst = max(worst, time.perf_counter() - t0)
        assert _name(sample_dataset, family) in [h.patient_name for h in hits[:3]], (q, [h.patient_name for h in hits])
    assert worst < 2.0
