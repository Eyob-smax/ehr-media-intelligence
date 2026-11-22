"""Deterministic, extractive summarizer used when no LLM is available.

It never generates language of its own: every phrase is copied from a record
section (chief complaint, assessment/diagnosis lists, imaging impressions,
flagged lab Observations), so it is grounded by construction. It exists so the
demo, the tests and CI work without an API key, and it gives a baseline to
compare LLM output against.
"""

from __future__ import annotations

import re

from .context import PatientContext, SourceItem
from .schema import MAX_WORDS, CitedItem, MediaItem, SummaryContent, word_count

_DX_HEADER = re.compile(r"^(discharge diagnoses|diagnoses|assessment)\s*:\s*$", re.IGNORECASE | re.MULTILINE)
_LIST_ITEM = re.compile(r"^\s*\d+\.\s*(.+?)\s*$")
_LAB_LINE = re.compile(r"^(?P<test>[^:\n]+):\s*(?P<val>[<>]?=?\d[\d.]*)\s*(?P<unit>[^(\n]*?)\s*\(ref (?P<ref>[^)]+)\)\s*"
                       r"\[(?P<flag>CRITICAL HIGH|CRITICAL LOW|HIGH|LOW|ABNORMAL)\]", re.MULTILINE)
_SAFETY = re.compile(r"(allerg[a-z]*:?\s*[^.]+|fall risk[^.]*|on anticoagulation|apixaban|rivaroxaban|warfarin)",
                     re.IGNORECASE)
_NORMAL_IMPRESSION = re.compile(r"^(no acute|normal|negative|unremarkable)", re.IGNORECASE)


def _diagnoses(items: list[SourceItem]) -> list[CitedItem]:
    seen: dict[str, CitedItem] = {}
    for it in items:
        if it.kind != "document":
            continue
        for m in _DX_HEADER.finditer(it.text):
            for line in it.text[m.end():].splitlines():
                if not line.strip():
                    continue
                lm = _LIST_ITEM.match(line)
                if not lm:
                    break
                key = lm.group(1).lower()
                if key in seen:
                    if it.sid not in seen[key].source_ids:
                        seen[key].source_ids.append(it.sid)
                else:
                    seen[key] = CitedItem(text=lm.group(1), source_ids=[it.sid])
    return list(seen.values())[:6]


def _chief_concern(ctx: PatientContext) -> str:
    encs = [i for i in ctx.items if i.kind == "encounter"]
    if encs:
        # Dated encounters always beat undated ones ("unknown" would otherwise sort after every ISO date).
        return max(encs, key=lambda e: (e.date != "unknown", e.date)).text
    for it in ctx.items:
        if m := re.search(r"(chief complaint|reason for visit):\s*(.+)", it.text, re.IGNORECASE):
            return m.group(2).rstrip(".")
    return ctx.items[0].title if ctx.items else "Not documented"


def _media(items: list[SourceItem]) -> list[MediaItem]:
    out = []
    for it in items:
        if it.kind not in ("imaging", "lab"):
            continue
        finding = it.conclusion or "See report"
        if it.kind == "lab" and finding.startswith("All reported"):
            finding = "Within reference range"
        out.append(MediaItem(source_id=it.sid, date=it.date, kind=it.kind, title=it.title, finding=finding))
        if len(out) == 4:
            break
    return out


def _anomalies(items: list[SourceItem]) -> list[CitedItem]:
    out: list[CitedItem] = []
    seen_tests: set[str] = set()
    for it in items:  # newest first, so the latest value of each test wins
        if it.kind == "lab":
            for m in _LAB_LINE.finditer(it.text):
                test = m["test"].strip()
                if test.lower() in seen_tests:
                    continue
                seen_tests.add(test.lower())
                out.append(CitedItem(
                    text=f"{test} {m['val']} {m['unit']} ({m['flag'].lower()}, ref {m['ref']}) on {it.date}".replace("  ", " "),
                    source_ids=[it.sid]))
        elif it.kind == "imaging" and it.conclusion and not _NORMAL_IMPRESSION.match(it.conclusion):
            out.append(CitedItem(text=f"{it.title}: {it.conclusion}", source_ids=[it.sid]))
        elif it.kind == "document":
            for m in _SAFETY.finditer(it.text):
                phrase = m.group(1).strip().rstrip(".")
                if phrase.lower() in ("apixaban", "rivaroxaban", "warfarin"):
                    phrase = f"Anticoagulation: {phrase.lower()}"
                if not any(phrase.lower() in a.text.lower() for a in out):
                    out.append(CitedItem(text=phrase[0].upper() + phrase[1:], source_ids=[it.sid]))
                break
    return out[:6]


class OfflineSummarizer:
    name = "offline"
    model = "rule-based-extractive-v2"

    def summarize(self, ctx: PatientContext) -> tuple[SummaryContent, str]:
        content = SummaryContent(
            chief_concern=_chief_concern(ctx),
            key_diagnoses=_diagnoses(ctx.items),
            recent_media=_media(ctx.items),
            flagged_anomalies=_anomalies(ctx.items),
        )
        # Trim lowest-priority items until the word budget is met.
        while word_count(content) > MAX_WORDS:
            for lst in (content.recent_media, content.flagged_anomalies, content.key_diagnoses):
                if len(lst) > 1:
                    lst.pop()
                    break
            else:
                break
        return content, self.model
