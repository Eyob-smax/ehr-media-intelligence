"""Validate Bundles against FHIR R4(B) and surface problems in a readable report.

Three layers, because the schema library alone is not enough:
1. **Structure** - every resource and the Bundle itself are parsed by the
   fhir.resources R4B Pydantic models (types, cardinality, required elements,
   primitive formats).
2. **Terminology** - fhir.resources does not enforce *required* value-set
   bindings (it accepts ``status: "bogus"``), so we check the status/gender
   codes whose bindings are ``required`` in the spec.
3. **Referential integrity** - every ``reference`` inside the Bundle must
   resolve to an entry ``fullUrl``, and every clinical resource must point at
   the Bundle's Patient. Dangling references are the most common real-world
   FHIR bug and the schema cannot see them.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

from pydantic import ValidationError

Severity = Literal["error", "warning"]

_MODEL_MODULES = {
    "Bundle": "bundle", "Patient": "patient", "Encounter": "encounter",
    "DocumentReference": "documentreference", "DiagnosticReport": "diagnosticreport",
    "Observation": "observation",
}

REQUIRED_CODES: dict[tuple[str, str], set[str]] = {
    ("Patient", "gender"): {"male", "female", "other", "unknown"},
    ("Encounter", "status"): {"planned", "arrived", "triaged", "in-progress", "onleave", "finished",
                              "cancelled", "entered-in-error", "unknown"},
    ("DocumentReference", "status"): {"current", "superseded", "entered-in-error"},
    ("DocumentReference", "docStatus"): {"preliminary", "final", "amended", "entered-in-error"},
    ("DiagnosticReport", "status"): {"registered", "partial", "preliminary", "final", "amended", "corrected",
                                     "appended", "cancelled", "entered-in-error", "unknown"},
    ("Observation", "status"): {"registered", "preliminary", "final", "amended", "corrected", "cancelled",
                                "entered-in-error", "unknown"},
    ("Bundle", "type"): {"document", "message", "transaction", "transaction-response", "batch",
                         "batch-response", "history", "searchset", "collection"},
}


def _model(resource_type: str):
    mod = _MODEL_MODULES.get(resource_type)
    if mod is None:
        return None
    return getattr(importlib.import_module(f"fhir.resources.R4B.{mod}"), resource_type)


@dataclass
class Issue:
    severity: Severity
    layer: Literal["structure", "terminology", "reference"]
    resource: str          # e.g. "DiagnosticReport/1234"
    path: str              # e.g. "status"
    message: str

    def as_dict(self) -> dict[str, str]:
        return self.__dict__.copy()


@dataclass
class BundleResult:
    patient_id: str
    bundle_id: str
    resource_counts: dict[str, int]
    issues: list[Issue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def as_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id, "bundle_id": self.bundle_id, "valid": self.valid,
            "resource_counts": self.resource_counts,
            "errors": sum(i.severity == "error" for i in self.issues),
            "warnings": sum(i.severity == "warning" for i in self.issues),
            "issues": [i.as_dict() for i in self.issues],
        }


def _pydantic_issues(exc: ValidationError, label: str) -> Iterator[Issue]:
    for err in exc.errors():
        path = ".".join(str(p) for p in err.get("loc", ())) or "(root)"
        yield Issue("error", "structure", label, path, err.get("msg", "invalid"))


def _walk_references(node: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(node, dict):
        if isinstance(node.get("reference"), str):
            yield path or "(root)", node["reference"]
        for k, v in node.items():
            yield from _walk_references(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk_references(v, f"{path}[{i}]")


def validate_bundle(bundle: dict[str, Any], patient_id: str = "") -> BundleResult:
    entries = bundle.get("entry") or []
    counts: dict[str, int] = {}
    for e in entries:
        rt = (e.get("resource") or {}).get("resourceType", "?")
        counts[rt] = counts.get(rt, 0) + 1
    result = BundleResult(patient_id=patient_id, bundle_id=str(bundle.get("id")),
                          resource_counts=dict(sorted(counts.items())))
    issues = result.issues

    # 1. Structure: the whole Bundle (which parses every contained resource) ...
    try:
        _model("Bundle").model_validate(bundle)
    except ValidationError as exc:
        issues.extend(_pydantic_issues(exc, f"Bundle/{bundle.get('id')}"))
    # ... and each resource on its own, so errors are attributed to a specific resource.
    for e in entries:
        res = e.get("resource") or {}
        label = f"{res.get('resourceType')}/{res.get('id')}"
        model = _model(res.get("resourceType", ""))
        if model is None:
            issues.append(Issue("warning", "structure", label, "resourceType", "no validator for this resource type"))
            continue
        try:
            model.model_validate(res)
        except ValidationError as exc:
            issues.extend(_pydantic_issues(exc, label))

    # 2. Terminology (required bindings).
    for (rt, fld), allowed in REQUIRED_CODES.items():
        targets = [bundle] if rt == "Bundle" else [e.get("resource") or {} for e in entries]
        for res in targets:
            if res.get("resourceType") != rt or fld not in res:
                continue
            if res[fld] not in allowed:
                issues.append(Issue("error", "terminology", f"{rt}/{res.get('id')}", fld,
                                    f"{res[fld]!r} is not in the required value set"))

    # 3. References.
    full_urls = {e.get("fullUrl") for e in entries}
    patients = [e for e in entries if (e.get("resource") or {}).get("resourceType") == "Patient"]
    if len(patients) != 1:
        issues.append(Issue("error", "reference", f"Bundle/{bundle.get('id')}", "entry",
                            f"expected exactly one Patient per bundle, found {len(patients)}"))
    patient_url = patients[0].get("fullUrl") if patients else None
    for e in entries:
        if not e.get("fullUrl"):
            issues.append(Issue("error", "reference", "Bundle", "entry.fullUrl", "entry is missing fullUrl"))
        res = e.get("resource") or {}
        label = f"{res.get('resourceType')}/{res.get('id')}"
        for path, target in _walk_references(res):
            if target not in full_urls:
                issues.append(Issue("error", "reference", label, path, f"dangling reference {target}"))
        if res.get("resourceType") != "Patient":
            subj = (res.get("subject") or {}).get("reference")
            if subj is None:
                issues.append(Issue("error", "reference", label, "subject", "clinical resource has no subject"))
            elif subj != patient_url:
                issues.append(Issue("error", "reference", label, "subject", "subject is not this bundle's Patient"))
        if res.get("resourceType") in ("DocumentReference", "DiagnosticReport") and not _has_encounter(res):
            issues.append(Issue("warning", "reference", label, "encounter", "no encounter context"))
    return result


def _has_encounter(res: dict) -> bool:
    if res.get("resourceType") == "DiagnosticReport":
        return "encounter" in res
    return bool((res.get("context") or {}).get("encounter"))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def build_report(results: list[BundleResult], names: dict[str, str] | None = None) -> dict[str, Any]:
    names = names or {}
    totals: dict[str, int] = {}
    for r in results:
        for k, v in r.resource_counts.items():
            totals[k] = totals.get(k, 0) + v
    return {
        "fhir_version": "4.0.1 (validated with fhir.resources R4B models)",
        "bundles": len(results),
        "valid_bundles": sum(r.valid for r in results),
        "total_errors": sum(i.severity == "error" for r in results for i in r.issues),
        "total_warnings": sum(i.severity == "warning" for r in results for i in r.issues),
        "resource_totals": dict(sorted(totals.items())),
        "results": [{**r.as_dict(), "patient": names.get(r.patient_id, "")} for r in results],
    }


def report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# FHIR Bundle Validation Report", "",
        f"- FHIR version: {report['fhir_version']}",
        f"- Bundles: **{report['valid_bundles']} / {report['bundles']} valid**",
        f"- Errors: {report['total_errors']} · Warnings: {report['total_warnings']}",
        "- Resources: " + ", ".join(f"{k} {v}" for k, v in report["resource_totals"].items()),
        "",
        "| Patient | Valid | Resources | Errors | Warnings |",
        "|---|---|---|---|---|",
    ]
    for r in report["results"]:
        lines.append(f"| {r['patient'] or r['patient_id']} | {'✅' if r['valid'] else '❌'} | "
                     f"{sum(r['resource_counts'].values())} | {r['errors']} | {r['warnings']} |")
    problems = [(r, i) for r in report["results"] for i in r["issues"]]
    lines += ["", "## Issues", ""]
    if not problems:
        lines.append("No issues found.")
    else:
        lines += ["| Severity | Layer | Patient | Resource | Path | Message |", "|---|---|---|---|---|---|"]
        for r, i in problems:
            lines.append(f"| {i['severity']} | {i['layer']} | {r['patient'] or r['patient_id']} | "
                         f"`{i['resource']}` | `{i['path']}` | {i['message']} |")
    return "\n".join(lines) + "\n"


def dumps(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2)
