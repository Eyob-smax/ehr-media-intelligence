"""Tests for AI summarization (Task 3).

The Claude path is exercised against a fake client so the suite never makes
network calls or needs an API key.
"""

import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from ehr.fhir.store import BundleStore
from ehr.summarize.cache import SummaryCache, cache_key
from ehr.summarize.claude import ClaudeSummarizer, SummarizerError
from ehr.summarize.context import build_context
from ehr.summarize.offline import OfflineSummarizer
from ehr.summarize.prompt import PROMPT_VERSION
from ehr.summarize.quality import check
from ehr.summarize.schema import DISCLAIMER, MAX_WORDS, CitedItem, MediaItem, SummaryContent, word_count
from ehr.summarize.service import SummaryService


@pytest.fixture()
def store(tmp_path, sample_bundles):
    s = BundleStore(tmp_path / "ehr.sqlite3")
    for pid, b in sample_bundles.items():
        s.save(pid, b)
    return s


@pytest.fixture()
def chf_patient(sample_dataset):
    return next(p.patient_id for p in sample_dataset.patients if p.family_name == "Quintero")


def _ctx(store, pid):
    return build_context(pid, store.get_bundle(pid))


# --- fake Claude client -------------------------------------------------------


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def fake_client(*responses):
    messages = FakeMessages(responses)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def reply(content: SummaryContent | dict, stop_reason="end_turn", model="claude-opus-5-5"):
    text = content.model_dump_json() if isinstance(content, SummaryContent) else json.dumps(content)
    return SimpleNamespace(stop_reason=stop_reason, stop_details=None, model=model,
                           content=[SimpleNamespace(type="text", text=text)])


def _grounded_content(ctx) -> SummaryContent:
    lab = next(i for i in ctx.items if i.kind == "lab" and "Potassium" in i.text)
    note = next(i for i in ctx.items if i.kind == "document" and "heart failure" in i.text.lower())
    return SummaryContent(
        chief_concern="Shortness of breath and bilateral leg swelling",
        key_diagnoses=[CitedItem(text="Acute on chronic systolic heart failure", source_ids=[note.sid])],
        recent_media=[MediaItem(source_id=lab.sid, date=lab.date, kind="lab", title=lab.title,
                                finding=lab.conclusion or "")],
        flagged_anomalies=[CitedItem(text="Potassium 5.4 mmol/L high", source_ids=[lab.sid])],
    )


# --- tests --------------------------------------------------------------------


def test_offline_summary_has_required_sections_disclaimer_and_word_limit(store):
    svc = SummaryService(store, SummaryCache(store.db_path), OfflineSummarizer())
    for s in svc.summarize_all():
        assert s.chief_concern
        assert s.word_count <= MAX_WORDS
        assert s.disclaimer == DISCLAIMER and "not a clinical decision" in s.disclaimer
        assert s.provenance.generator == "offline"
        assert s.confidence.level == "high", s.quality_issues


def test_offline_summary_flags_abnormal_labs_and_cites_records(store, chf_patient):
    s = SummaryService(store, SummaryCache(store.db_path), OfflineSummarizer()).summarize(chf_patient)
    anomalies = " ".join(a.text for a in s.flagged_anomalies)
    assert "BNP 1250" in anomalies
    assert any("heart failure" in d.text.lower() for d in s.key_diagnoses)
    assert all(a.source_ids for a in s.flagged_anomalies + s.key_diagnoses)


def test_cache_prevents_redundant_generation(store, chf_patient):
    calls = []

    class Counting(OfflineSummarizer):
        def summarize(self, ctx):
            calls.append(ctx.patient_id)
            return super().summarize(ctx)

    svc = SummaryService(store, SummaryCache(store.db_path), Counting())
    first = svc.summarize(chf_patient)
    second = svc.summarize(chf_patient)
    assert calls == [chf_patient]
    assert first.model_dump() == second.model_dump()
    assert svc.stats["cache_hits"] == 1


def test_cache_key_changes_with_records_prompt_and_model():
    base = cache_key("p", "hash1", PROMPT_VERSION, "m")
    assert base != cache_key("p", "hash2", PROMPT_VERSION, "m")
    assert base != cache_key("p", "hash1", "other-prompt", "m")
    assert base != cache_key("p", "hash1", PROMPT_VERSION, "m2")


def test_record_change_invalidates_cache(store, chf_patient):
    calls = []

    class Counting(OfflineSummarizer):
        def summarize(self, ctx):
            calls.append(1)
            return super().summarize(ctx)

    svc = SummaryService(store, SummaryCache(store.db_path), Counting())
    svc.summarize(chf_patient)
    bundle = store.get_bundle(chf_patient)
    bundle["entry"][0]["resource"]["gender"] = "unknown"  # any change to the records
    store.save(chf_patient, bundle)
    svc.summarize(chf_patient)
    assert len(calls) == 2


def test_claude_summarizer_sends_structured_output_request(store, chf_patient):
    ctx = _ctx(store, chf_patient)
    client, msgs = fake_client(reply(_grounded_content(ctx)))
    content, model = ClaudeSummarizer("claude-opus-5-5", client=client).summarize(ctx)

    call = msgs.calls[0]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["fallbacks"] == "default"
    assert "<patient_records>" in call["messages"][0]["content"]
    assert content.key_diagnoses[0].text == "Acute on chronic systolic heart failure"
    assert check(content, ctx).grounding_ratio == 1.0


def test_claude_over_length_summary_triggers_one_shorten_request(store, chf_patient):
    ctx = _ctx(store, chf_patient)
    good = _grounded_content(ctx)
    long = good.model_copy(update={"chief_concern": "word " * (MAX_WORDS + 10)})
    client, msgs = fake_client(reply(long), reply(good))
    content, _ = ClaudeSummarizer("m", client=client).summarize(ctx)
    assert len(msgs.calls) == 2
    assert "over the 200-word limit" in msgs.calls[1]["messages"][-1]["content"]
    assert word_count(content) <= MAX_WORDS


def test_refusal_and_bad_json_raise_summarizer_error(store, chf_patient):
    ctx = _ctx(store, chf_patient)
    client, _ = fake_client(reply({}, stop_reason="refusal"))
    with pytest.raises(SummarizerError, match="declined"):
        ClaudeSummarizer("m", client=client).summarize(ctx)
    client, _ = fake_client(reply({"chief_concern": 5}))
    with pytest.raises(SummarizerError, match="schema"):
        ClaudeSummarizer("m", client=client).summarize(ctx)


def test_auth_failure_falls_back_to_offline_and_stops_calling_the_api(store):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    auth_error = anthropic.AuthenticationError("invalid x-api-key", response=httpx2.Response(401, request=req),
                                               body=None)
    client, msgs = fake_client(auth_error)
    svc = SummaryService(store, SummaryCache(store.db_path), ClaudeSummarizer("m", client=client))
    results = svc.summarize_all()

    assert len(msgs.calls) == 1, "a permanent error disables the LLM for the rest of the run"
    assert all(s.provenance.generator == "offline" for s in results)
    assert "authentication" in results[0].provenance.fallback_reason
    assert svc.stats["fallbacks"] == 1


def test_grounding_check_catches_hallucinated_numbers_and_bad_citations(store, chf_patient):
    ctx = _ctx(store, chf_patient)
    content = _grounded_content(ctx)
    lab_sid = content.flagged_anomalies[0].source_ids[0]
    content.flagged_anomalies = [
        CitedItem(text="Potassium 6.9 mmol/L high", source_ids=[lab_sid]),     # wrong number
        CitedItem(text="Pulmonary embolism", source_ids=["R999"]),              # nonexistent record
        CitedItem(text="Metastatic pancreatic carcinoma", source_ids=[lab_sid]),  # unsupported by cited text
        CitedItem(text="Atrial fibrillation", source_ids=[]),                   # uncited
    ]
    rep = check(content, ctx)
    assert rep.checked - rep.supported == 4
    joined = " ".join(rep.issues)
    assert "6.9" in joined and "R999" in joined and "no citation" in joined and "weakly supported" in joined
