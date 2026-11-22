"""Command-line entry point.

    python -m ehr.cli build      # ingest -> FHIR -> validate -> store -> summarize -> index
    python -m ehr.cli ingest     # Task 1 only, writes clean dataset + audit log
    python -m ehr.cli serve      # API + UI on http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date

from .config import Settings, get_settings
from .fhir.mapping import build_bundles
from .fhir.store import BundleStore, write_bundle_files
from .fhir.validation import build_report, dumps, report_markdown, validate_bundle
from .ingest.pipeline import run_ingestion, write_outputs
from .summarize.cache import SummaryCache
from .summarize.service import SummaryService, make_summarizer

log = logging.getLogger("ehr")


def _step(name: str):
    class _Timer:
        def __enter__(self):
            self.t = time.perf_counter()
            print(f"▶ {name} ...", flush=True)
            return self

        def __exit__(self, *exc):
            if not exc[0]:
                print(f"  done in {time.perf_counter() - self.t:.2f}s", flush=True)
    return _Timer()


def ensure_sample_data(s: Settings, patients: int) -> None:
    """Generate the synthetic dataset when the raw directory is empty (first run)."""
    import importlib.util
    import sys

    from .config import ROOT

    with _step(f"No raw data found; generating synthetic dataset (16 curated + {patients} cohort patients)"):
        spec = importlib.util.spec_from_file_location("generate_data", ROOT / "scripts" / "generate_data.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        print("  " + json.dumps(mod.build(s.raw_dir, seed=7, cohort=patients)))


def cmd_ingest(s: Settings, reference: date | None):
    with _step("Task 1 · ingest and clean"):
        ds = run_ingestion(s.raw_dir, reference=reference)
        paths = write_outputs(ds, s.out_dir)
        print("  " + json.dumps(ds.stats()))
        print(f"  audit log: {paths['audit']}")
    return ds


def cmd_fhir(s: Settings, ds) -> BundleStore:
    with _step("Task 2 · map to FHIR R4, validate, store"):
        bundles = build_bundles(ds)
        results = [validate_bundle(b, pid) for pid, b in bundles.items()]
        report = build_report(results, {p.patient_id: p.display_name for p in ds.patients})
        (s.out_dir / "validation_report.json").write_text(dumps(report))
        (s.out_dir / "validation_report.md").write_text(report_markdown(report))
        write_bundle_files(bundles, s.out_dir / "bundles")
        store = BundleStore(s.db_path)
        for r in results:
            store.save(r.patient_id, bundles[r.patient_id], valid=r.valid,
                       issues=[i.as_dict() for i in r.issues])
        if removed := store.prune(set(bundles)):
            print(f"  removed {removed} stale patient(s) no longer in the dataset (e.g. merged away)")
        print(f"  {report['valid_bundles']}/{report['bundles']} bundles valid · "
              f"{report['total_errors']} errors · {report['total_warnings']} warnings · "
              f"resources {report['resource_totals']}")
        print(f"  report: {s.out_dir / 'validation_report.md'}")
    return store


def cmd_summarize(s: Settings, store: BundleStore, force: bool) -> SummaryService:
    summarizer = make_summarizer(s)
    with _step(f"Task 3 · clinical summaries ({summarizer.name}: {summarizer.model})"):
        svc = SummaryService(store, SummaryCache(s.db_path), summarizer)
        summaries = svc.summarize_all(force=force)
        out = s.out_dir / "summaries"
        out.mkdir(parents=True, exist_ok=True)
        for sm in summaries:
            (out / f"{sm.patient_id}.json").write_text(sm.model_dump_json(indent=2))
        levels = {lvl: sum(sm.confidence.level == lvl for sm in summaries) for lvl in ("high", "medium", "low")}
        print(f"  {len(summaries)} summaries · generated {svc.stats['generated']} · cache hits "
              f"{svc.stats['cache_hits']} · fallbacks {svc.stats['fallbacks']}")
        print(f"  confidence {levels} · max {max((sm.word_count for sm in summaries), default=0)} words · "
              f"{sum(bool(sm.quality_issues) for sm in summaries)} with quality issues")
    return svc


def cmd_index(s: Settings, store: BundleStore, svc: SummaryService) -> None:
    from .search.embeddings import get_embedder
    from .search.index import SearchIndex

    embedder = get_embedder(s)
    with _step(f"Task 4 · embed and index ({embedder.name})"):
        index = SearchIndex(s.chroma_dir, embedder, mrn_resolver=store.patient_for_mrn)
        summaries = {p["patient_id"]: svc.latest(p["patient_id"]) for p in store.list_patients()}
        n = index.rebuild(store.iter_bundles(), {k: v for k, v in summaries.items() if v})
        r = index.search("chest x-ray findings")
        print(f"  {n} chunks indexed · sample query returned {len(r.hits)} hits in {r.took_ms:.0f} ms")


def cmd_serve(host: str, port: int) -> None:
    import uvicorn

    print(f"Serving UI and API on http://{host}:{port}  (API docs at /docs)")
    uvicorn.run("ehr.api:create_app", factory=True, host=host, port=port)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ehr", description="EHR Media Intelligence pipeline")
    ap.add_argument("command", choices=["ingest", "build", "serve"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reference-date", type=date.fromisoformat, default=None,
                    help="'today' for date sanity checks (defaults to the real current date)")
    ap.add_argument("--force-summaries", action="store_true", help="ignore the summary cache and regenerate")
    ap.add_argument("--patients", type=int, default=300,
                    help="cohort size when generating sample data on first run (data/raw empty)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    s = get_settings()
    if args.command == "serve":
        cmd_serve(args.host, args.port)
        return
    s.out_dir.mkdir(parents=True, exist_ok=True)
    if not s.raw_dir.exists() or not any(s.raw_dir.iterdir()):
        ensure_sample_data(s, args.patients)
    ds = cmd_ingest(s, args.reference_date)
    if args.command == "ingest":
        return
    store = cmd_fhir(s, ds)
    svc = cmd_summarize(s, store, args.force_summaries)
    cmd_index(s, store, svc)


if __name__ == "__main__":
    main()
