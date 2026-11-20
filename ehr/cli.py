"""Command-line entry point.

    python -m ehr.cli build      # ingest -> FHIR -> validate -> store (-> summarize -> index)
    python -m ehr.cli ingest     # Task 1 only, writes clean dataset + audit log
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
        print(f"  {report['valid_bundles']}/{report['bundles']} bundles valid · "
              f"{report['total_errors']} errors · {report['total_warnings']} warnings · "
              f"resources {report['resource_totals']}")
        print(f"  report: {s.out_dir / 'validation_report.md'}")
    return store


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="ehr", description="EHR Media Intelligence pipeline")
    ap.add_argument("command", choices=["ingest", "build"])
    ap.add_argument("--reference-date", type=date.fromisoformat, default=None,
                    help="'today' for date sanity checks (defaults to the real current date)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    s = get_settings()
    s.out_dir.mkdir(parents=True, exist_ok=True)
    ds = cmd_ingest(s, args.reference_date)
    if args.command == "ingest":
        return
    cmd_fhir(s, ds)


if __name__ == "__main__":
    main()
