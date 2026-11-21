"""Orchestrates summary generation: cache lookup -> summarizer -> quality checks -> cache write."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Protocol

from ..config import Settings
from ..fhir.store import BundleStore, record_hash
from .cache import SummaryCache, cache_key
from .claude import ClaudeSummarizer, SummarizerError, credentials_available
from .context import PatientContext, build_context
from .offline import OfflineSummarizer
from .prompt import PROMPT_VERSION
from .quality import check, confidence, enforce_length
from .schema import ClinicalSummary, Provenance, SummaryContent

log = logging.getLogger(__name__)


class Summarizer(Protocol):
    name: str
    model: str

    def summarize(self, ctx: PatientContext) -> tuple[SummaryContent, str]: ...


def make_summarizer(settings: Settings) -> Summarizer:
    mode = settings.summarizer.lower()
    if mode == "claude" or (mode == "auto" and credentials_available()):
        return ClaudeSummarizer(settings.claude_model)
    if mode not in ("auto", "offline"):
        raise ValueError(f"unknown EHR_SUMMARIZER={settings.summarizer!r} (use auto, claude or offline)")
    return OfflineSummarizer()


class SummaryService:
    def __init__(self, store: BundleStore, cache: SummaryCache, summarizer: Summarizer):
        self.store = store
        self.cache = cache
        self.primary = summarizer
        self.offline = OfflineSummarizer()
        self._llm_disabled_reason: str | None = None
        self.stats = {"cache_hits": 0, "generated": 0, "fallbacks": 0}

    def summarize(self, patient_id: str, *, force: bool = False) -> ClinicalSummary:
        bundle = self.store.get_bundle(patient_id)
        if bundle is None:
            raise KeyError(patient_id)
        rhash = record_hash(bundle)
        key = cache_key(patient_id, rhash, PROMPT_VERSION, self.primary.model)
        if not force and (hit := self.cache.get(key)):
            self.stats["cache_hits"] += 1
            return hit
        # A previous run may have fallen back to offline; reuse that too rather than re-hitting a broken API.
        offline_key = cache_key(patient_id, rhash, PROMPT_VERSION, self.offline.model)
        if not force and self._llm_disabled_reason and (hit := self.cache.get(offline_key)):
            self.stats["cache_hits"] += 1
            return hit

        ctx = build_context(patient_id, bundle)
        summarizer: Summarizer = self.primary
        fallback_reason = None
        if self._llm_disabled_reason and summarizer is not self.offline:
            summarizer, fallback_reason = self.offline, self._llm_disabled_reason
        try:
            content, served_by = summarizer.summarize(ctx)
        except SummarizerError as e:
            log.warning("LLM summary failed for %s (%s); using offline summarizer", patient_id, e)
            if e.permanent:
                self._llm_disabled_reason = str(e)
            summarizer, fallback_reason = self.offline, str(e)
            content, served_by = self.offline.summarize(ctx)
            self.stats["fallbacks"] += 1

        content, trimmed = enforce_length(content)
        rep = check(content, ctx)
        if trimmed:
            rep.issues.append("trimmed to fit the 200-word limit")
        summary = ClinicalSummary(
            **content.model_dump(),
            patient_id=patient_id,
            record_hash=rhash,
            word_count=rep.word_count,
            confidence=confidence(rep, generator=summarizer.name),
            provenance=Provenance(generator=summarizer.name, model=served_by, prompt_version=PROMPT_VERSION,  # type: ignore[arg-type]
                                  fallback_reason=fallback_reason),
            generated_at=datetime.now(timezone.utc),
            quality_issues=rep.issues,
        )
        self.cache.put(cache_key(patient_id, rhash, PROMPT_VERSION, summarizer.model), summary)
        self.stats["generated"] += 1
        return summary

    def summarize_all(self, *, force: bool = False) -> list[ClinicalSummary]:
        return [self.summarize(p["patient_id"], force=force) for p in self.store.list_patients()]

    def latest(self, patient_id: str) -> ClinicalSummary | None:
        bundle = self.store.get_bundle(patient_id)
        return self.cache.latest_for(patient_id, record_hash(bundle) if bundle else None)
