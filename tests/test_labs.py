"""Tests for the lab catalog: aliases, unit conversion, messy values, flags, vendor CSV layouts."""

from datetime import date, datetime

import pytest

from ehr.ingest.labs import convert, find_panel, find_test, normalize_flag, parse_lab_value
from ehr.ingest.normalize import derive_flag, parse_datetime
from ehr.ingest.pipeline import run_ingestion


@pytest.mark.parametrize("alias, key", [
    ("HGB", "hgb"), ("Hb", "hgb"), ("Potassium, serum", "k"), ("CREAT", "cr"), ("Glucose, random", "glu"),
    ("HbA1c", "a1c"), ("Troponin I", "trop"), ("SGPT", "alt"), ("D-Dimer", "ddimer"),
])
def test_test_aliases_resolve_to_catalog(alias, key):
    assert find_test(alias).key == key


def test_panel_aliases_resolve():
    assert find_panel("CBC W/O DIFF").name == "CBC"
    assert find_panel("Basic Metabolic Panel").loinc == "51990-0"
    assert find_panel("unknown battery") is None


@pytest.mark.parametrize("key, value, unit, expected", [
    ("glu", 11.8, "mmol/L", 212.6),       # glucose mmol/L -> mg/dL
    ("cr", 186, "umol/L", 2.1),           # creatinine umol/L -> mg/dL (rounded, no invented precision)
    ("hgb", 98, "g/L", 9.8),              # hemoglobin g/L -> g/dL
    ("a1c", 79, "mmol/mol", 9.38),        # IFCC -> NGSP %
    ("trop", 840, "ng/L", 0.84),
])
def test_unit_conversion_to_canonical(key, value, unit, expected):
    from ehr.ingest.labs import TEST_BY_KEY

    out, canon, note = convert(TEST_BY_KEY[key], value, unit)
    assert out == pytest.approx(expected, rel=1e-3)
    assert canon == TEST_BY_KEY[key].unit and "converted" in note


@pytest.mark.parametrize("raw, value, cmp_, flag", [
    ("5.6", 5.6, None, None), ("5,6", 5.6, None, None), ("1,250", 1250, None, None),
    ("<0.01", 0.01, "<", None), (">500", 500, ">", None), ("H 5.6", 5.6, None, "H"), ("5.6HH", 5.6, None, "HH"),
    ("pending", None, None, None), ("hemolyzed", None, None, None), ("", None, None, None),
])
def test_messy_lab_values_parse(raw, value, cmp_, flag):
    pv = parse_lab_value(raw)
    assert (pv.value, pv.comparator, pv.flag_hint) == (value, cmp_, flag)


def test_critical_flags_survive_and_abnormal_is_refined():
    assert normalize_flag("Critical High") == "HH"
    assert derive_flag(7.1, "3.5-5.1", "HH") == ("HH", None)
    flag, note = derive_flag(2.9, "3.5-5.1", "A")
    assert flag == "L" and "refined" in note


def test_dotted_and_compact_dates():
    assert parse_datetime("01.03.2025 10:30").value == datetime(2025, 3, 1, 10, 30)
    assert parse_datetime("202503011030").value == datetime(2025, 3, 1, 10, 30)
    assert parse_datetime("20250301").value.date() == date(2025, 3, 1)


def test_vendor_csv_layout_latin1_semicolons_and_units(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    body = ("PatientID;Patient Name;BirthDate;ResultDateTime;OrderName;Analyte;Result;Units;RefRange;AbnFlag\n"
            "MRN 0000042;Núñez, José;06.05.1970;01.03.2025 10:00;CHEM7;GLU;11,8;mmol/L;3,9-5,5;H\n"
            "MRN 0000042;Núñez, José;06.05.1970;01.03.2025 10:00;CHEM7;K;7,1;mmol/L;3,5-5,1;HH\n"
            "MRN 0000042;Núñez, José;06.05.1970;01.03.2025 10:00;CHEM7;CREAT;pending;umol/L;53-115;\n"
            "broken;row\n")
    (raw / "vendor_b.csv").write_bytes(body.encode("latin-1"))
    ds = run_ingestion(raw, reference=date(2025, 10, 1))

    p = ds.patients[0]
    assert (p.given_name, p.family_name) == ("José", "Núñez")
    doc = ds.documents[0]
    assert doc.title == "BMP Results"
    obs = {o.test: o for o in doc.observations}
    assert obs["Glucose"].unit == "mg/dL" and obs["Glucose"].value == pytest.approx(212.6, rel=1e-3)
    assert obs["Glucose"].ref_range == "70-99", "source range was in mmol/L, replaced with canonical"
    assert obs["Potassium"].flag == "HH"
    assert obs["Creatinine"].value is None
    reasons = " ".join(e.reason for e in ds.all_audit())
    assert "latin-1" in reasons and "expected 10 columns" in reasons and "converted 11.8 mmol/L" in reasons
