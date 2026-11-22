"""Chroma-backed semantic index over FHIR records and AI summaries.

Indexing
- Each DocumentReference / DiagnosticReport is split into ~120-word chunks
  (MiniLM truncates at 256 word pieces, so long notes would otherwise lose
  their tail). Each chunk is prefixed with the record title and category so
  short chunks still carry context.
- Each patient's AI summary is indexed as a ``Patient``-level record, so a
  query like "heart failure with high potassium" can match the synthesis even
  when no single document says all of it.
- Metadata carries patient, resource type and an integer ``date_int``
  (YYYYMMDD) because Chroma range filters only work on numbers.

Ranking
- Semantic score = cosine similarity from Chroma.
- A small lexical boost (query terms present in the chunk, title or patient
  name) helps exact clinical terms, names and MRNs, which embeddings handle
  poorly. Final score = 0.8 * semantic + 0.2 * lexical.
- AI-summary hits are multiplied by 0.9 so that, at equal relevance, the
  primary record outranks the summary derived from it.
- Chunks are collapsed to one hit per resource (best chunk wins), top-k returned.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..fhir.text import bundle_patient, bundle_records, patient_display
from ..ingest.normalize import normalize_mrn
from ..summarize.schema import ClinicalSummary, render_text
from .embeddings import Embedder
from .synonyms import expand, group_matches

RESOURCE_TYPES = ("DocumentReference", "DiagnosticReport", "Patient")
CHUNK_WORDS = 120
CHUNK_OVERLAP = 30
SEMANTIC_WEIGHT = 0.8
SUMMARY_PENALTY = 0.9

_MRN_IN_QUERY = re.compile(r"\b(?:mrn[\s:#-]*)?(\d{3,7})\b", re.IGNORECASE)


def chunk_words(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    out, step = [], size - overlap
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return out


def _date_int(d: str | None) -> int:
    return int(d.replace("-", "")[:8]) if d else 0


@dataclass
class SearchHit:
    score: float
    semantic: float
    lexical: float
    patient_id: str
    patient_name: str
    mrn: str
    resource_type: str
    resource_id: str
    title: str
    category: str
    date: str | None
    snippet: str


@dataclass
class SearchResponse:
    hits: list[SearchHit]
    took_ms: float
    candidates: int
    filters: dict[str, Any] = field(default_factory=dict)


class SearchIndex:
    def __init__(self, chroma_dir: Path, embedder: Embedder, collection: str = "ehr_records",
                 mrn_resolver: Callable[[str], str | None] | None = None):
        self.embedder = embedder
        self.mrn_resolver = mrn_resolver
        Path(chroma_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(chroma_dir),
                                                settings=ChromaSettings(anonymized_telemetry=False))
        # Namespacing by embedder avoids mixing vectors of different models/dimensions.
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", embedder.name)[-40:]
        self.collection_name = f"{collection}__{safe}"
        self.col = self._collection()

    def _collection(self):
        return self.client.get_or_create_collection(
            self.collection_name, metadata={"hnsw:space": "cosine"}, embedding_function=None)

    def count(self) -> int:
        return self.col.count()

    # ------------------------------------------------------------------ build

    def rebuild(self, bundles: Iterable[tuple[str, dict]], summaries: dict[str, ClinicalSummary] | None = None) -> int:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass
        self.col = self._collection()
        summaries = summaries or {}

        ids, docs, metas = [], [], []
        for pid, bundle in bundles:
            patient = bundle_patient(bundle)
            name, mrn = patient_display(patient)
            base = {"patient_id": pid, "patient_name": name, "mrn": mrn}
            latest = None
            for rec in bundle_records(bundle):
                latest = max(latest or "", rec.date or "")
                body = rec.text if not rec.conclusion or rec.conclusion in rec.text else f"{rec.conclusion}\n{rec.text}"
                for n, chunk in enumerate(chunk_words(body)):
                    ids.append(f"{rec.resource_id}:{n}")
                    docs.append(f"{rec.title} ({rec.category}). {chunk}")
                    metas.append({**base, "resource_type": rec.resource_type, "resource_id": rec.resource_id,
                                  "title": rec.title, "category": rec.category, "date": rec.date or "",
                                  "date_int": _date_int(rec.date), "chunk": n})
            if (s := summaries.get(pid)) is not None:
                for n, chunk in enumerate(chunk_words(render_text(s))):
                    ids.append(f"{patient['id']}:summary:{n}")
                    docs.append(f"AI clinical summary. {chunk}")
                    metas.append({**base, "resource_type": "Patient", "resource_id": patient["id"],
                                  "title": "AI clinical summary", "category": "Summary", "date": latest or "",
                                  "date_int": _date_int(latest), "chunk": n})

        for start in range(0, len(ids), 256):
            sl = slice(start, start + 256)
            self.col.add(ids=ids[sl], documents=docs[sl], metadatas=metas[sl],
                         embeddings=self.embedder.embed(docs[sl]))
        return len(ids)

    # ------------------------------------------------------------------ query

    def search(self, query: str, *, top_k: int = 5, resource_types: list[str] | None = None,
               date_from: date | None = None, date_to: date | None = None) -> SearchResponse:
        t0 = time.perf_counter()
        clauses: list[dict] = []
        if resource_types:
            clauses.append({"resource_type": {"$in": list(resource_types)}})
        if date_from or date_to:
            clauses.append({"date_int": {"$gt": 0}})  # undated records can't satisfy a date range
        if date_from:
            clauses.append({"date_int": {"$gte": _date_int(date_from.isoformat())}})
        if date_to:
            clauses.append({"date_int": {"$lte": _date_int(date_to.isoformat())}})
        mrn, explicit = self._mrn_from_query(query)
        patient_id = self._resolve_mrn(mrn) if mrn else None
        filters = {"resource_types": resource_types or [], "date_from": date_from and date_from.isoformat(),
                   "date_to": date_to and date_to.isoformat(), "mrn": mrn if patient_id else None}
        if patient_id:
            clauses.append({"patient_id": patient_id})
        elif explicit:
            # The user asked for a specific MRN ("MRN 1234") that doesn't exist: say so rather than guess.
            return SearchResponse([], round((time.perf_counter() - t0) * 1000, 1), 0,
                                  {**filters, "mrn": mrn, "mrn_not_found": True})
        else:
            mrn = None  # a bare number that isn't a known MRN (e.g. "2025") is just a search term
        where = None if not clauses else clauses[0] if len(clauses) == 1 else {"$and": clauses}

        total = self.col.count()
        if total == 0:
            return SearchResponse([], (time.perf_counter() - t0) * 1000, 0, filters)
        n = min(total, max(top_k * 8, 40))
        embed_text, groups = expand(query)
        res = self.col.query(query_embeddings=self.embedder.embed([embed_text]), n_results=n, where=where,
                             include=["documents", "metadatas", "distances"])
        q_terms = {p for g in groups for p in g}
        best: dict[str, SearchHit] = {}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            semantic = max(0.0, 1.0 - float(dist))
            if mrn:
                lexical = 1.0  # exact identifier match
            else:
                hay = f"{doc} {meta['patient_name']}".lower()
                lexical = sum(group_matches(g, hay) for g in groups) / len(groups) if groups else 0.0
            score = SEMANTIC_WEIGHT * semantic + (1 - SEMANTIC_WEIGHT) * lexical
            if meta["resource_type"] == "Patient":
                score *= SUMMARY_PENALTY  # prefer the primary record over the AI summary derived from it
            rid = meta["resource_id"]
            if rid in best and best[rid].score >= score:
                continue
            best[rid] = SearchHit(
                score=round(score, 4), semantic=round(semantic, 4), lexical=round(lexical, 4),
                patient_id=meta["patient_id"], patient_name=meta["patient_name"], mrn=meta["mrn"],
                resource_type=meta["resource_type"], resource_id=rid, title=meta["title"],
                category=meta["category"], date=meta["date"] or None, snippet=_snippet(doc, q_terms),
            )
        hits = sorted(best.values(), key=lambda h: h.score, reverse=True)[:top_k]
        return SearchResponse(hits, round((time.perf_counter() - t0) * 1000, 1), len(res["ids"][0]), filters)

    @staticmethod
    def _mrn_from_query(query: str) -> tuple[str | None, bool]:
        """Return (candidate MRN, explicit?). Explicit means the user wrote "MRN ..."; a bare number is only a candidate."""
        q = query.strip()
        if m := re.search(r"\bmrn[\s:#-]*(\d{1,7})\b", q, re.IGNORECASE):
            explicit = True
        elif m := _MRN_IN_QUERY.fullmatch(q):
            explicit = False
        else:
            return None, False
        try:
            return normalize_mrn(m.group(1)), explicit
        except ValueError:
            return None, False

    def _resolve_mrn(self, mrn: str) -> str | None:
        """Map an MRN (official or merged/old) to a patient id via the injected resolver, else the index metadata."""
        if self.mrn_resolver is not None:
            return self.mrn_resolver(mrn)
        got = self.col.get(where={"mrn": mrn}, limit=1, include=["metadatas"])
        return got["metadatas"][0]["patient_id"] if got["metadatas"] else None


def _snippet(doc: str, q_terms: set[str], width: int = 260) -> str:
    """Window of the chunk around the first query term, so cards show *why* a record matched."""
    text = " ".join(doc.split())
    low = text.lower()
    positions = [low.find(t) for t in q_terms if low.find(t) >= 0]
    start = max(0, min(positions) - 60) if positions else 0
    snippet = text[start:start + width]
    return ("…" if start > 0 else "") + snippet + ("…" if start + width < len(text) else "")
