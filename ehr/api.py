"""FastAPI application: search, patient detail and FHIR retrieval endpoints, plus the static UI.

Run with ``python -m ehr.cli serve`` (or ``uvicorn ehr.api:create_app --factory``).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import ROOT, Settings, get_settings
from .fhir.store import BundleStore
from .fhir.text import bundle_patient
from .search.embeddings import Embedder, get_embedder
from .search.index import SearchIndex
from .summarize.cache import SummaryCache
from .summarize.schema import ClinicalSummary
from .summarize.service import SummaryService, make_summarizer

log = logging.getLogger(__name__)

ResourceType = Literal["DocumentReference", "DiagnosticReport", "Patient"]


# ----------------------------------------------------------------------------- schemas


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500, description="Free-text clinical question or keywords")
    top_k: int = Field(5, ge=1, le=20)


class SearchResult(BaseModel):
    rank: int
    score: float = Field(description="Blended relevance, 0-1 (0.8 semantic + 0.2 keyword)")
    semantic_score: float
    keyword_score: float
    patient_id: str
    patient_name: str
    mrn: str
    resource_type: str
    resource_id: str
    title: str
    category: str
    date: str | None
    snippet: str
    summary_snippet: str | None = None


class SearchResponseModel(BaseModel):
    query: str
    took_ms: float
    total: int
    filters: dict
    results: list[SearchResult]


class LinkedResource(BaseModel):
    resource_id: str
    resource_type: str
    category: str | None
    title: str | None
    date: str | None
    conclusion: str | None
    preview: str


class PatientDetail(BaseModel):
    patient_id: str
    name: str
    mrn: str
    alternate_mrns: list[str]
    gender: str | None
    birth_date: str | None
    bundle_valid: bool
    validation_issues: list[dict]
    summary: ClinicalSummary | None
    resources: list[LinkedResource]


# ----------------------------------------------------------------------------- app state


class AppState:
    def __init__(self, settings: Settings, embedder: Embedder | None = None):
        self.settings = settings
        self.store = BundleStore(settings.db_path)
        self.summaries = SummaryService(self.store, SummaryCache(settings.db_path), make_summarizer(settings))
        self.embedder = embedder or get_embedder(settings)
        self.index = SearchIndex(settings.chroma_dir, self.embedder)
        self._snippets: dict[str, str] = {}

    def warm_up(self) -> None:
        """Load the embedding model and make sure the index exists, so the first query is fast."""
        self.embedder.embed(["warm up"])
        if self.store.count() and self.index.count() == 0:
            log.info("search index empty; building from stored bundles")
            self.index.rebuild(self.store.iter_bundles(), self.all_summaries())
        self.refresh_snippets()

    def all_summaries(self) -> dict[str, ClinicalSummary]:
        out = {}
        for p in self.store.list_patients():
            s = self.summaries.latest(p["patient_id"]) or self.summaries.summarize(p["patient_id"])
            out[p["patient_id"]] = s
        return out

    def refresh_snippets(self) -> None:
        self._snippets = {pid: snippet(s) for pid, s in self.all_summaries().items()}

    def snippet_for(self, pid: str) -> str | None:
        return self._snippets.get(pid)


def snippet(s: ClinicalSummary, limit: int = 180) -> str:
    dx = "; ".join(d.text for d in s.key_diagnoses[:2])
    text = f"{s.chief_concern}." + (f" Dx: {dx}." if dx else "")
    if s.flagged_anomalies:
        text += f" ⚑ {s.flagged_anomalies[0].text}"
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# ----------------------------------------------------------------------------- factory


def create_app(settings: Settings | None = None, *, embedder: Embedder | None = None,
               frontend_dir: Path | None = ROOT / "frontend") -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = AppState(settings, embedder)
        state.warm_up()
        app.state.ehr = state
        yield

    app = FastAPI(
        title="EHR Media Intelligence API",
        version="0.1.0",
        description="Semantic search over FHIR R4 patient records with AI-generated clinical summaries. "
                    "Synthetic data only. Summaries are AI-generated and not clinical decisions.",
        lifespan=lifespan,
    )
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    def st(request: Request) -> AppState:
        return request.app.state.ehr

    @app.get("/health", tags=["meta"])
    def health(request: Request):
        s = st(request)
        return {"status": "ok", "patients": s.store.count(), "indexed_chunks": s.index.count(),
                "embedder": s.embedder.name, "summarizer": s.summaries.primary.name}

    @app.post("/search", response_model=SearchResponseModel, tags=["search"])
    def search(
        body: SearchRequest,
        request: Request,
        resource_type: list[ResourceType] | None = Query(None, description="Filter by FHIR resource type (repeatable)"),
        date_from: date | None = Query(None, description="Earliest record date (inclusive), YYYY-MM-DD"),
        date_to: date | None = Query(None, description="Latest record date (inclusive), YYYY-MM-DD"),
    ):
        if date_from and date_to and date_from > date_to:
            raise HTTPException(422, "date_from must be on or before date_to")
        s = st(request)
        r = s.index.search(body.query, top_k=body.top_k, resource_types=resource_type,
                           date_from=date_from, date_to=date_to)
        results = [SearchResult(rank=i, score=h.score, semantic_score=h.semantic, keyword_score=h.lexical,
                                patient_id=h.patient_id, patient_name=h.patient_name, mrn=h.mrn,
                                resource_type=h.resource_type, resource_id=h.resource_id, title=h.title,
                                category=h.category, date=h.date, snippet=h.snippet,
                                summary_snippet=s.snippet_for(h.patient_id))
                   for i, h in enumerate(r.hits, start=1)]
        return SearchResponseModel(query=body.query, took_ms=r.took_ms, total=len(results), filters=r.filters,
                                   results=results)

    @app.get("/patients", tags=["patients"])
    def list_patients(request: Request):
        return st(request).store.list_patients()

    @app.get("/patients/{patient_id}", response_model=PatientDetail, tags=["patients"])
    def patient_detail(patient_id: str, request: Request):
        s = st(request)
        bundle = s.store.get_bundle(patient_id)
        meta = s.store.get_meta(patient_id)
        if bundle is None or meta is None:
            raise HTTPException(404, f"patient {patient_id} not found")
        pat = bundle_patient(bundle)
        summary = s.summaries.latest(patient_id) or s.summaries.summarize(patient_id)
        resources = [LinkedResource(resource_id=r["resource_id"], resource_type=r["resource_type"],
                                    category=r["category"], title=r["title"], date=r["date"],
                                    conclusion=r["conclusion"], preview=" ".join((r["text"] or "").split())[:280])
                     for r in s.store.resources_for(patient_id)]
        return PatientDetail(
            patient_id=patient_id, name=meta["patient_name"], mrn=meta["mrn"],
            alternate_mrns=[i["value"] for i in pat.get("identifier", []) if i.get("use") == "old"],
            gender=pat.get("gender"), birth_date=pat.get("birthDate"), bundle_valid=meta["valid"],
            validation_issues=meta["issues"], summary=summary, resources=resources,
        )

    @app.get("/patients/{patient_id}/bundle", tags=["fhir"])
    def patient_bundle(patient_id: str, request: Request):
        b = st(request).store.get_bundle(patient_id)
        if b is None:
            raise HTTPException(404, f"patient {patient_id} not found")
        return b

    @app.get("/resources/{resource_id}", tags=["fhir"])
    def get_resource(resource_id: str, request: Request):
        r = st(request).store.get_resource(resource_id)
        if r is None:
            raise HTTPException(404, f"resource {resource_id} not found")
        return r

    if frontend_dir and Path(frontend_dir).is_dir():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="ui")
    return app
