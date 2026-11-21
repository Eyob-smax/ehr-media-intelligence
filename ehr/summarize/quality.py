"""Automated quality checks for generated summaries.

These run on every summary (LLM or offline) and drive the confidence field:

- **Completeness**: chief concern present, required sections are lists.
- **Length**: under the 200-word budget (trimmed deterministically if not).
- **Citations**: every cited id exists in the patient context.
- **Grounding**: each diagnosis/anomaly shares its key terms with the cited
  record(s), and every number in it appears verbatim in the cited text.
  Numbers are the highest-risk hallucination in clinical summaries (a wrong
  potassium value is worse than a missing one), so a mismatched number is
  always treated as unsupported.

Confidence is derived from these checks, never self-reported by the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .context import PatientContext
from .schema import MAX_WORDS, CitedItem, Confidence, SummaryContent, word_count

_STOP = set("""
a an and are as at be by for from has have in into is it its of on or the to was were with without
no not this that these those than then there their his her patient patients status post due left right
mild moderate severe acute chronic new consistent seen noted findings finding report results result
""".split())
_TOKEN = re.compile(r"[a-z][a-z0-9-]+")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")


def _terms(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if len(t) >= 4 and t not in _STOP}


def _numbers(text: str) -> set[str]:
    return {n.rstrip("0").rstrip(".") if "." in n else n for n in _NUMBER.findall(text)}


@dataclass
class QualityReport:
    word_count: int
    issues: list[str] = field(default_factory=list)
    supported: int = 0
    checked: int = 0

    @property
    def grounding_ratio(self) -> float:
        return 1.0 if self.checked == 0 else self.supported / self.checked


def _support(item: CitedItem, ctx: PatientContext, label: str, issues: list[str]) -> bool:
    by_id = ctx.by_id()
    if not item.source_ids:
        issues.append(f"{label} has no citation: {item.text!r}")
        return False
    unknown = [s for s in item.source_ids if s not in by_id]
    if unknown:
        issues.append(f"{label} cites unknown record(s) {unknown}: {item.text!r}")
        return False
    source = " ".join(f"{by_id[s].title} {by_id[s].text} {by_id[s].conclusion or ''} {by_id[s].date}"
                      for s in item.source_ids)
    missing_numbers = _numbers(item.text) - _numbers(source)
    if missing_numbers:
        issues.append(f"{label} has numbers not found in cited records {sorted(missing_numbers)}: {item.text!r}")
        return False
    terms = _terms(item.text)
    if terms:
        overlap = len(terms & _terms(source)) / len(terms)
        if overlap < 0.5:
            issues.append(f"{label} weakly supported by {item.source_ids} ({overlap:.0%} term overlap): {item.text!r}")
            return False
    return True


def check(content: SummaryContent, ctx: PatientContext) -> QualityReport:
    rep = QualityReport(word_count=word_count(content))
    if not content.chief_concern.strip():
        rep.issues.append("chief concern is empty")
    if rep.word_count > MAX_WORDS:
        rep.issues.append(f"summary is {rep.word_count} words (limit {MAX_WORDS})")

    for label, items in (("diagnosis", content.key_diagnoses), ("anomaly", content.flagged_anomalies)):
        for item in items:
            rep.checked += 1
            rep.supported += _support(item, ctx, label, rep.issues)

    by_id = ctx.by_id()
    for m in content.recent_media:
        rep.checked += 1
        src = by_id.get(m.source_id)
        if src is None:
            rep.issues.append(f"media item cites unknown record {m.source_id!r}")
        elif src.date != m.date:
            rep.issues.append(f"media item {m.source_id} date {m.date} does not match record date {src.date}")
        else:
            rep.supported += _support(CitedItem(text=m.finding, source_ids=[m.source_id]), ctx, "media finding",
                                      rep.issues)
    return rep


def enforce_length(content: SummaryContent) -> tuple[SummaryContent, bool]:
    """Last-resort trim: drop lowest-priority items until under the budget. Returns (content, trimmed?)."""
    c = content.model_copy(deep=True)
    trimmed = False
    while word_count(c) > MAX_WORDS:
        for lst in (c.recent_media, c.flagged_anomalies, c.key_diagnoses):
            if len(lst) > 1:
                lst.pop()
                trimmed = True
                break
        else:
            break
    return c, trimmed


def confidence(rep: QualityReport, *, generator: str) -> Confidence:
    score = rep.grounding_ratio
    reasons = [f"{rep.supported}/{rep.checked} items grounded in cited records"]
    if rep.word_count > MAX_WORDS:
        score -= 0.2
        reasons.append("over word limit")
    if generator == "offline":
        reasons.append("extractive offline summary (no LLM); phrasing copied from source records")
    score = max(0.0, min(1.0, score))
    level = "high" if score >= 0.9 else "medium" if score >= 0.7 else "low"
    return Confidence(level=level, score=round(score, 2), reasons=reasons)
