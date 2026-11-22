"""Laboratory catalog: test/panel aliases, LOINC codes, canonical units and conversions.

Real lab feeds name the same analyte a dozen ways ("HGB", "Hb", "Hemoglobin,
whole blood") and report it in different units depending on the instrument
and country (glucose in mg/dL or mmol/L, creatinine in mg/dL or umol/L).
Everything is mapped to one canonical name, LOINC code and unit here so that
trends, flags and summaries compare like with like.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LabTest:
    key: str
    name: str
    loinc: str
    unit: str
    ref_low: float | None
    ref_high: float | None
    aliases: tuple[str, ...]
    # source unit (lower-case, normalized) -> (factor, offset); canonical = value * factor + offset
    conversions: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def ref_range(self) -> str:
        if self.ref_low is not None and self.ref_high is not None:
            return f"{self.ref_low:g}-{self.ref_high:g}"
        if self.ref_high is not None:
            return f"<{self.ref_high:g}"
        if self.ref_low is not None:
            return f">{self.ref_low:g}"
        return ""


_SAME = (1.0, 0.0)

TESTS: list[LabTest] = [
    LabTest("hgb", "Hemoglobin", "718-7", "g/dL", 12.0, 17.5, ("hgb", "hb", "hemoglobin", "haemoglobin"),
            {"g/l": (0.1, 0), "mmol/l": (1.611, 0)}),
    LabTest("wbc", "WBC", "6690-2", "10*3/uL", 4.0, 11.0,
            ("wbc", "white blood cell count", "white blood cells", "leukocytes", "wbc count"),
            {"10^9/l": _SAME, "k/ul": _SAME, "x10e3/ul": _SAME, "10*9/l": _SAME}),
    LabTest("plt", "Platelets", "777-3", "10*3/uL", 150, 400, ("plt", "platelets", "platelet count"),
            {"10^9/l": _SAME, "k/ul": _SAME, "x10e3/ul": _SAME, "10*9/l": _SAME}),
    LabTest("na", "Sodium", "2951-2", "mmol/L", 135, 145, ("na", "sodium", "sodium serum", "sodium, serum"),
            {"meq/l": _SAME}),
    LabTest("k", "Potassium", "2823-3", "mmol/L", 3.5, 5.1, ("k", "potassium", "potassium serum", "potassium, serum", "pot"),
            {"meq/l": _SAME}),
    LabTest("cr", "Creatinine", "2160-0", "mg/dL", 0.6, 1.3, ("cr", "creat", "creatinine", "creatinine, serum"),
            {"umol/l": (1 / 88.4, 0), "µmol/l": (1 / 88.4, 0)}),
    LabTest("glu", "Glucose", "2345-7", "mg/dL", 70, 99, ("glu", "glucose", "glucose fasting", "glucose, random", "glucose random"),
            {"mmol/l": (18.016, 0)}),
    LabTest("a1c", "Hemoglobin A1c", "4548-4", "%", 4.0, 5.6, ("a1c", "hba1c", "hemoglobin a1c", "glycated hemoglobin"),
            {"mmol/mol": (0.09148, 2.152)}),
    LabTest("ldl", "LDL Cholesterol", "13457-7", "mg/dL", None, 99, ("ldl", "ldl cholesterol", "ldl-c", "ldl calc"),
            {"mmol/l": (38.67, 0)}),
    LabTest("hdl", "HDL Cholesterol", "2085-9", "mg/dL", 40, 100, ("hdl", "hdl cholesterol", "hdl-c"),
            {"mmol/l": (38.67, 0)}),
    LabTest("chol", "Total Cholesterol", "2093-3", "mg/dL", None, 199, ("chol", "cholesterol", "total cholesterol"),
            {"mmol/l": (38.67, 0)}),
    LabTest("tsh", "TSH", "3016-3", "mIU/L", 0.4, 4.0, ("tsh", "thyrotropin", "thyroid stimulating hormone"),
            {"uiu/ml": _SAME, "miu/l": _SAME}),
    LabTest("bnp", "BNP", "30934-4", "pg/mL", None, 100, ("bnp", "b-type natriuretic peptide"), {"ng/l": _SAME}),
    LabTest("trop", "Troponin I", "10839-9", "ng/mL", None, 0.04, ("trop", "troponin", "troponin i", "tni", "trop i"),
            {"ng/l": (0.001, 0)}),
    LabTest("lipase", "Lipase", "3040-3", "U/L", 10, 60, ("lipase",), {"iu/l": _SAME}),
    LabTest("lactate", "Lactate", "2524-7", "mmol/L", 0.5, 2.0, ("lactate", "lactic acid"), {"mg/dl": (0.111, 0)}),
    LabTest("inr", "INR", "6301-6", "", 0.8, 1.2, ("inr", "pt/inr", "international normalized ratio"), {}),
    LabTest("alt", "ALT", "1742-6", "U/L", 7, 56, ("alt", "sgpt", "alanine aminotransferase"), {"iu/l": _SAME}),
    LabTest("ast", "AST", "1920-8", "U/L", 10, 40, ("ast", "sgot", "aspartate aminotransferase"), {"iu/l": _SAME}),
    LabTest("ddimer", "D-dimer", "48065-7", "mg/L FEU", None, 0.5, ("d-dimer", "ddimer", "d dimer"),
            {"ug/ml feu": _SAME, "ng/ml feu": (0.001, 0), "mg/l": _SAME}),
    LabTest("ferritin", "Ferritin", "2276-4", "ng/mL", 15, 300, ("ferritin",), {"ug/l": _SAME}),
]


@dataclass(frozen=True)
class Panel:
    key: str
    name: str
    loinc: str
    loinc_display: str
    aliases: tuple[str, ...]
    tests: tuple[str, ...]


PANELS: list[Panel] = [
    Panel("cbc", "CBC", "58410-2", "CBC panel - Blood by Automated count",
          ("cbc", "cbc w/o diff", "cbc with differential", "complete blood count", "hemogram"), ("hgb", "wbc", "plt")),
    Panel("bmp", "BMP", "51990-0", "Basic metabolic panel - Blood",
          ("bmp", "basic metabolic panel", "chem7", "chem 7", "renal panel"), ("na", "k", "cr", "glu")),
    Panel("lipid", "Lipid Panel", "24331-1", "Lipid panel - Serum or Plasma",
          ("lipid panel", "lipids", "lipid profile", "fasting lipid panel"), ("ldl", "hdl", "chol")),
    Panel("a1c", "HbA1c", "4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood",
          ("hba1c", "a1c", "hemoglobin a1c", "glycohemoglobin"), ("a1c",)),
    Panel("tsh", "TSH", "3016-3", "Thyrotropin [Units/volume] in Serum or Plasma", ("tsh", "thyroid screen"), ("tsh",)),
    Panel("bnp", "BNP", "30934-4", "Natriuretic peptide B [Mass/volume] in Serum or Plasma", ("bnp",), ("bnp",)),
    Panel("trop", "Troponin", "10839-9", "Troponin I.cardiac [Mass/volume] in Serum or Plasma",
          ("troponin", "troponin i", "trop"), ("trop",)),
    Panel("hepatic", "Hepatic Function Panel", "24325-3", "Hepatic function panel - Serum or Plasma",
          ("hepatic function panel", "lft", "lfts", "liver panel"), ("alt", "ast")),
    Panel("lipase", "Lipase", "3040-3", "Lipase [Enzymatic activity/volume] in Serum or Plasma", ("lipase",), ("lipase",)),
    Panel("lactate", "Lactate", "2524-7", "Lactate [Moles/volume] in Serum or Plasma", ("lactate", "lactic acid"),
          ("lactate",)),
    Panel("coag", "PT/INR", "6301-6", "INR in Platelet poor plasma by Coagulation assay",
          ("pt/inr", "inr", "coagulation panel", "coags"), ("inr",)),
    Panel("ddimer", "D-dimer", "48065-7", "Fibrin D-dimer FEU [Mass/volume] in Platelet poor plasma",
          ("d-dimer", "ddimer"), ("ddimer",)),
    Panel("ferritin", "Ferritin", "2276-4", "Ferritin [Mass/volume] in Serum or Plasma", ("ferritin", "iron studies"),
          ("ferritin",)),
]


def _norm(s: str) -> str:
    return " ".join(s.lower().replace("_", " ").split())


_TEST_BY_ALIAS = {_norm(a): t for t in TESTS for a in (t.aliases + (t.name, t.key))}
_PANEL_BY_ALIAS = {_norm(a): p for p in PANELS for a in (p.aliases + (p.name, p.key))}
TEST_BY_KEY = {t.key: t for t in TESTS}


def find_test(name: str | None) -> LabTest | None:
    if not name:
        return None
    n = _norm(name)
    return _TEST_BY_ALIAS.get(n) or _TEST_BY_ALIAS.get(re.sub(r"[,(].*$", "", n).strip())


def find_panel(name: str | None) -> Panel | None:
    if not name:
        return None
    return _PANEL_BY_ALIAS.get(_norm(name))


def _unit_key(u: str) -> str:
    return _norm(u).replace("μ", "µ").replace("mcg", "ug")


def convert(test: LabTest, value: float, unit: str | None) -> tuple[float, str, str | None]:
    """Return (value, canonical unit, note). Unknown units are left as-is with a note."""
    if not unit or _unit_key(unit) == _unit_key(test.unit):
        return value, test.unit or (unit or ""), None
    conv = test.conversions.get(_unit_key(unit))
    if conv is None:
        return value, unit, f"unrecognized unit {unit!r} for {test.name}; value kept as reported"
    factor, offset = conv
    raw = value * factor + offset
    # Do not invent precision the source never had: 2 decimals below 10, 1 decimal above.
    out = round(raw, 2 if abs(raw) < 10 else 1)
    return out, test.unit, f"converted {value:g} {unit} to {out:g} {test.unit}"


# ---------------------------------------------------------------------------
# Values and flags
# ---------------------------------------------------------------------------

NON_NUMERIC = ("pending", "cancelled", "canceled", "hemolyzed", "haemolysed", "see comment", "tnp", "qns",
               "not performed", "specimen clotted")

_VALUE = re.compile(r"^(?P<flag1>HH|LL|H|L|A)?\s*(?P<cmp><=|>=|<|>)?\s*(?P<num>\d[\d.,]*)\s*(?P<flag2>HH|LL|H|L|A)?\*?$",
                    re.IGNORECASE)


@dataclass
class ParsedValue:
    value: float | None
    comparator: str | None = None
    flag_hint: str | None = None
    note: str | None = None


def parse_lab_value(raw: object) -> ParsedValue:
    """Parse real-world result strings: ``5.6``, ``5,6``, ``1,250``, ``<0.01``, ``H 5.6``, ``5.6H``, ``pending``."""
    if raw is None:
        return ParsedValue(None)
    s = str(raw).strip()
    if not s:
        return ParsedValue(None)
    low = s.lower()
    if any(w in low for w in NON_NUMERIC):
        return ParsedValue(None, note=f"non-numeric result {s!r}")
    m = _VALUE.match(s)
    if not m:
        return ParsedValue(None, note=f"unparseable result {s!r}")
    num, note = m["num"], None
    if "," in num and "." not in num:
        whole, _, frac = num.partition(",")
        if len(frac) == 3 and num.count(",") >= 1 and all(len(p) == 3 for p in num.split(",")[1:]):
            num, note = num.replace(",", ""), f"comma in {s!r} read as thousands separator"
        else:
            num, note = num.replace(",", "."), f"comma in {s!r} read as decimal separator"
    else:
        num = num.replace(",", "")
    try:
        value = float(num)
    except ValueError:
        return ParsedValue(None, note=f"unparseable result {s!r}")
    flag = (m["flag1"] or m["flag2"] or "").upper() or None
    cmp_ = m["cmp"]
    if cmp_:
        note = (note + "; " if note else "") + f"censored result {cmp_}{value:g}"
    return ParsedValue(value, cmp_, flag, note)


FLAG_CODES = {"H", "L", "HH", "LL", "A", "N"}


def normalize_flag(raw: object) -> str | None:
    """HL7 v2 table 0078 subset. ``>``/``<`` style or words are mapped; unknown flags become None."""
    if raw is None:
        return None
    s = str(raw).strip().upper()
    s = {"HIGH": "H", "LOW": "L", "CRIT HIGH": "HH", "CRITICAL HIGH": "HH", "CRIT LOW": "LL", "CRITICAL LOW": "LL",
         "ABN": "A", "ABNORMAL": "A", "NORMAL": "N", "": ""}.get(s, s)
    return s if s in FLAG_CODES else None
