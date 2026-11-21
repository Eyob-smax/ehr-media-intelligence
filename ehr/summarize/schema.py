"""Summary data model.

``SummaryContent`` is what the LLM (or the offline summarizer) produces. Every
diagnosis, media item and anomaly must cite the record ids it came from, which
is what makes automated grounding checks possible.

``ClinicalSummary`` wraps the content with everything the application adds
after generation: word count, grounding-based confidence, the disclaimer and
provenance (which generator, model and prompt version).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

DISCLAIMER = (
    "AI-generated summary for informational purposes only. It is not a clinical decision or a substitute "
    "for review of the source record by a qualified clinician."
)
MAX_WORDS = 200


class CitedItem(BaseModel):
    text: str
    source_ids: list[str] = Field(default_factory=list)


class MediaItem(BaseModel):
    source_id: str
    date: str
    kind: Literal["imaging", "lab", "document"]
    title: str
    finding: str


class SummaryContent(BaseModel):
    chief_concern: str
    key_diagnoses: list[CitedItem] = Field(default_factory=list)
    recent_media: list[MediaItem] = Field(default_factory=list)
    flagged_anomalies: list[CitedItem] = Field(default_factory=list)


class Confidence(BaseModel):
    level: Literal["high", "medium", "low"]
    score: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)


class Provenance(BaseModel):
    generator: Literal["claude", "offline"]
    model: str | None = None
    prompt_version: str
    fallback_reason: str | None = None


class ClinicalSummary(SummaryContent):
    patient_id: str
    record_hash: str
    word_count: int
    confidence: Confidence
    disclaimer: str = DISCLAIMER
    provenance: Provenance
    generated_at: datetime
    quality_issues: list[str] = Field(default_factory=list)


# JSON schema for Claude structured outputs (kept explicit rather than generated so it stays within
# the supported subset: objects with required keys and additionalProperties false).
_CITED = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "source_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["text", "source_ids"],
    "additionalProperties": False,
}
CONTENT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "chief_concern": {"type": "string"},
        "key_diagnoses": {"type": "array", "items": _CITED},
        "recent_media": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "date": {"type": "string"},
                    "kind": {"type": "string", "enum": ["imaging", "lab", "document"]},
                    "title": {"type": "string"},
                    "finding": {"type": "string"},
                },
                "required": ["source_id", "date", "kind", "title", "finding"],
                "additionalProperties": False,
            },
        },
        "flagged_anomalies": {"type": "array", "items": _CITED},
    },
    "required": ["chief_concern", "key_diagnoses", "recent_media", "flagged_anomalies"],
    "additionalProperties": False,
}


def render_text(c: SummaryContent) -> str:
    """Plain-text rendering used for display, word counting and embedding."""
    parts = [f"Chief concern: {c.chief_concern}"]
    if c.key_diagnoses:
        parts.append("Key diagnoses: " + "; ".join(d.text for d in c.key_diagnoses))
    if c.recent_media:
        parts.append("Recent media: " + "; ".join(f"{m.date} {m.title}: {m.finding}" for m in c.recent_media))
    if c.flagged_anomalies:
        parts.append("Flagged anomalies: " + "; ".join(a.text for a in c.flagged_anomalies))
    return "\n".join(parts)


def word_count(c: SummaryContent) -> int:
    return len(render_text(c).split())
