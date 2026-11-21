"""Unit tests for the pure normalization functions (Task 1)."""

from datetime import date, datetime

import pytest

from ehr.ingest.models import Gender
from ehr.ingest.normalize import (
    clean_text,
    derive_flag,
    fingerprint,
    normalize_gender,
    normalize_mrn,
    parse_date,
    parse_datetime,
    split_name,
)

REF = date(2026, 1, 15)


@pytest.mark.parametrize("raw", ["1234", "0001234", "mrn1234", "MRN 0001234", "MRN-0001234", " mrn: 1234 "])
def test_mrn_variants_normalize_to_one_canonical_form(raw):
    assert normalize_mrn(raw) == "MRN-0001234"


@pytest.mark.parametrize("raw", [None, "", "   ", "MRN-12A4", "123456789"])
def test_invalid_mrns_are_rejected_not_guessed(raw):
    with pytest.raises(ValueError):
        normalize_mrn(raw)


@pytest.mark.parametrize("raw, expected", [
    ("1958-02-14", date(1958, 2, 14)),
    ("02/14/1958", date(1958, 2, 14)),
    ("14 Feb 1958", date(1958, 2, 14)),
    ("1958/2/14", date(1958, 2, 14)),
    ("February 14, 1958", date(1958, 2, 14)),
    ("14-Feb-1958", date(1958, 2, 14)),
])
def test_inconsistent_date_formats_parse_to_the_same_date(raw, expected):
    assert parse_date(raw, reference=REF)[0] == expected


def test_two_digit_birth_year_never_lands_in_the_future():
    # "58" must be 1958, not 2058; "01" is 2001 because that is not after the reference date.
    assert parse_date("2/14/58", reference=REF, allow_future=False)[0] == date(1958, 2, 14)
    born_2001, note = parse_date("3/15/01", reference=REF, allow_future=False)
    assert born_2001 == date(2001, 3, 15)
    assert "two-digit year" in note


def test_two_digit_year_uses_sliding_window_for_non_birth_dates():
    assert parse_date("6/1/30", reference=REF)[0] == date(2030, 6, 1)
    assert parse_date("6/1/90", reference=REF)[0] == date(1990, 6, 1)


def test_ambiguous_day_month_is_flagged_and_unambiguous_dmy_is_detected():
    d, note = parse_date("04/07/1961", reference=REF)
    assert d == date(1961, 4, 7) and "ambiguous" in note
    d, note = parse_date("13/04/1961", reference=REF)
    assert d == date(1961, 4, 13) and "day/month" in note


def test_future_birth_date_is_rejected():
    with pytest.raises(ValueError, match="future"):
        parse_date("2030-01-01", reference=REF, allow_future=False)


def test_datetime_keeps_time_and_reports_it():
    p = parse_datetime("2024-11-28T10:30:00Z", reference=REF)
    assert p.value == datetime(2024, 11, 28, 10, 30) and p.has_time
    assert not parse_datetime("2024-11-28", reference=REF).has_time


@pytest.mark.parametrize("raw, expected", [
    ("M", Gender.male), ("male", Gender.male), ("MALE", Gender.male), ("1", Gender.male),
    ("F", Gender.female), ("Female", Gender.female), ("2", Gender.female),
    ("U", Gender.unknown), ("", Gender.unknown), ("X", Gender.other),
])
def test_gender_codes_map_to_fhir_administrative_gender(raw, expected):
    assert normalize_gender(raw)[0] is expected


def test_unrecognized_gender_becomes_unknown_with_a_note():
    g, note = normalize_gender("purple")
    assert g is Gender.unknown and "unrecognized" in note


@pytest.mark.parametrize("raw, expected", [
    ("QUINTERO, MARISOL", ("Marisol", "Quintero")),
    ("Priya Raman", ("Priya", "Raman")),
    ("  van  der  Berg,  ANNA  ", ("Anna", "Van Der Berg")),
    ("o'brien-mcdonald, jo", ("Jo", "O'Brien-McDonald")),
    ("", (None, None)),
])
def test_name_parsing(raw, expected):
    assert split_name(raw) == expected


def test_ocr_cleanup_rejoins_hyphenation_and_soft_wraps_but_keeps_sections():
    raw = "XR CHEST\nFindings: Interval   improve-\nment of right lower\nlobe opacity.\nImpression: Improving pneumonia.\n\f"
    assert clean_text(raw) == (
        "XR CHEST\nFindings: Interval improvement of right lower lobe opacity.\nImpression: Improving pneumonia."
    )


def test_fingerprint_ignores_whitespace_and_case():
    assert fingerprint("Chest  pain.\n", "X") == fingerprint("chest pain.", "x")
    assert fingerprint("chest pain") != fingerprint("chest pains")


@pytest.mark.parametrize("title, kind", [
    ("Outside Radiology Report", "imaging_report"),
    ("CT Head without Contrast", "imaging_report"),
    ("Direct Admission Note", "scanned_document"),
    ("Conflict of Interest Form", "scanned_document"),
    ("Discharge Summary", "discharge_summary"),
    ("Office Visit Note", "progress_note"),
])
def test_scanned_title_classification_matches_whole_words(title, kind):
    from ehr.ingest.loaders import _kind_from_title

    assert _kind_from_title(title).value == kind


def test_lab_flag_is_derived_when_missing_and_corrected_when_wrong():
    assert derive_flag(132, "135-145", "")[0] == "L"
    flag, note = derive_flag(4.2, "3.5-5.1", "H")
    assert flag == "N" and "disagreed" in note
    assert derive_flag(131, "<100", "H") == ("H", None)
    assert derive_flag(None, "135-145", "") == (None, None)
