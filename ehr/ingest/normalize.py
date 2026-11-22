"""Pure normalization functions.

Each function returns the canonical value plus an optional note explaining any
judgment call (ambiguous date, inferred century, unrecognized code). The caller
turns those notes into audit entries. Keeping these pure makes them trivial to
unit test.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from typing import NamedTuple

from .models import Gender

# ---------------------------------------------------------------------------
# MRN
# ---------------------------------------------------------------------------

MRN_DIGITS = 7
_MRN_PREFIX = re.compile(r"^\s*(mrn)?[\s:#-]*", re.IGNORECASE)


def normalize_mrn(raw: object) -> str:
    """Return a canonical ``MRN-0001234`` string.

    Accepts ``1234``, ``0001234``, ``mrn1234``, ``MRN 0001234``, ``MRN-0001234``.
    Raises ``ValueError`` for empty values, letters inside the number, or more
    than seven significant digits, since guessing an MRN is unsafe.
    """
    if raw is None:
        raise ValueError("MRN is missing")
    s = str(raw).strip()
    if not s:
        raise ValueError("MRN is empty")
    body = _MRN_PREFIX.sub("", s)
    if not body.isdigit():
        raise ValueError(f"MRN {s!r} contains non-numeric characters")
    digits = body.lstrip("0") or "0"
    if len(digits) > MRN_DIGITS:
        raise ValueError(f"MRN {s!r} has more than {MRN_DIGITS} significant digits")
    return f"MRN-{int(digits):0{MRN_DIGITS}d}"


def mrn_to_patient_id(mrn: str) -> str:
    """Deterministic, opaque patient id. Stable across runs so caches and URLs survive re-ingestion."""
    return "pat-" + hashlib.sha256(mrn.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------


class Parsed(NamedTuple):
    value: datetime
    note: str | None
    has_time: bool


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

_RE_ISO = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[T ](\d{1,2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?$")
_RE_YMD_SLASH = re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})$")
_RE_SLASH = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})(?:\s+(\d{1,2}):(\d{2}))?$")
_RE_TEXT_MDY = re.compile(r"^([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})$")
_RE_TEXT_DMY = re.compile(r"^(\d{1,2})[\s-]([A-Za-z]{3,9})\.?[\s-](\d{4})$")
_RE_DOTTED_DMY = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})(?:\s+(\d{1,2}):(\d{2}))?$")  # European: 01.03.2025
_RE_COMPACT = re.compile(r"^(\d{4})(\d{2})(\d{2})(?:(\d{2})(\d{2})(\d{2})?)?$")      # HL7-style: 20250301[1030[00]]


def _month(name: str) -> int:
    m = _MONTHS.get(name[:3].lower())
    if not m:
        raise ValueError(f"unknown month {name!r}")
    return m


def expand_two_digit_year(yy: int, reference: date, allow_future: bool) -> tuple[int, str]:
    """Pick the century for a 2-digit year.

    For birth dates (``allow_future=False``) the year must not be in the future,
    so ``85`` becomes 1985 and ``24`` becomes 2024 only if that is not after
    ``reference``. For other dates we use a +/-50-year sliding window.
    """
    century = reference.year // 100 * 100
    candidate = century + yy
    if allow_future:
        if candidate - reference.year > 50:
            candidate -= 100
    elif candidate > reference.year:
        candidate -= 100
    return candidate, f"two-digit year {yy:02d} interpreted as {candidate}"


def parse_datetime(raw: object, *, reference: date | None = None, allow_future: bool = True) -> Parsed:
    """Parse the date formats seen across our sources into a naive datetime.

    Ambiguous ``a/b/yyyy`` values are read as US month/day unless the first
    number cannot be a month, in which case day/month is used. Either way the
    returned note records the interpretation.
    """
    if raw is None:
        raise ValueError("date is missing")
    s = str(raw).strip()
    if not s:
        raise ValueError("date is empty")
    ref = reference or date.today()
    note: str | None = None

    if m := _RE_ISO.match(s):
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
        hh, mm, ss = int(m[4] or 0), int(m[5] or 0), int(m[6] or 0)
        if m[7] and m[7] != "Z":
            note = f"timezone offset {m[7]} dropped"
        result = Parsed(datetime(y, mo, d, hh, mm, ss), note, m[4] is not None)
    elif m := _RE_YMD_SLASH.match(s):
        result = Parsed(datetime(int(m[1]), int(m[2]), int(m[3])), None, False)
    elif m := _RE_SLASH.match(s):
        a, b, y = int(m[1]), int(m[2]), int(m[3])
        if len(m[3]) == 2:
            y, note = expand_two_digit_year(y, ref, allow_future)
        if a > 12 and b <= 12:
            mo, d = b, a
            note = _join(note, "read as day/month (first field > 12)")
        else:
            mo, d = a, b
            if a <= 12 and b <= 12 and a != b:
                note = _join(note, "ambiguous; read as US month/day")
        hh, mm = int(m[4] or 0), int(m[5] or 0)
        result = Parsed(datetime(y, mo, d, hh, mm), note, m[4] is not None)
    elif m := _RE_TEXT_MDY.match(s):
        result = Parsed(datetime(int(m[3]), _month(m[1]), int(m[2])), None, False)
    elif m := _RE_TEXT_DMY.match(s):
        result = Parsed(datetime(int(m[3]), _month(m[2]), int(m[1])), None, False)
    elif m := _RE_DOTTED_DMY.match(s):
        result = Parsed(datetime(int(m[3]), int(m[2]), int(m[1]), int(m[4] or 0), int(m[5] or 0)),
                        "dotted date read as day.month.year", m[4] is not None)
    elif m := _RE_COMPACT.match(s):
        result = Parsed(datetime(int(m[1]), int(m[2]), int(m[3]), int(m[4] or 0), int(m[5] or 0), int(m[6] or 0)),
                        None, m[4] is not None)
    else:
        raise ValueError(f"unrecognized date format: {s!r}")

    if not allow_future and result.value.date() > ref:
        raise ValueError(f"date {s!r} is in the future")
    return result


def parse_date(raw: object, *, reference: date | None = None, allow_future: bool = True) -> tuple[date, str | None]:
    p = parse_datetime(raw, reference=reference, allow_future=allow_future)
    return p.value.date(), p.note


def _join(a: str | None, b: str) -> str:
    return f"{a}; {b}" if a else b


# ---------------------------------------------------------------------------
# Gender
# ---------------------------------------------------------------------------

_GENDER_MAP = {
    "m": Gender.male, "male": Gender.male, "man": Gender.male,
    "f": Gender.female, "female": Gender.female, "woman": Gender.female,
    "o": Gender.other, "other": Gender.other, "x": Gender.other, "nonbinary": Gender.other,
    "u": Gender.unknown, "unknown": Gender.unknown, "unk": Gender.unknown, "": Gender.unknown,
    # ISO/IEC 5218 numeric codes
    "0": Gender.unknown, "1": Gender.male, "2": Gender.female, "9": Gender.unknown,
}


def normalize_gender(raw: object) -> tuple[Gender, str | None]:
    """Map free-text and coded gender to FHIR administrative-gender."""
    if raw is None:
        return Gender.unknown, "gender missing"
    key = str(raw).strip().lower()
    if key in _GENDER_MAP:
        g = _GENDER_MAP[key]
        note = "ISO/IEC 5218 numeric code" if key.isdigit() else None
        return g, note
    return Gender.unknown, f"unrecognized gender code {raw!r}"


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


def smart_title(s: str) -> str:
    """Title-case a name while keeping Mc/Mac, O', and hyphenated parts sensible."""
    def cap(part: str) -> str:
        w = part.lower().capitalize()
        if w.startswith("Mc") and len(w) > 2:
            w = "Mc" + w[2:].capitalize()
        return w

    def fix(word: str) -> str:
        return "-".join(cap(p) for p in word.split("-"))
    out = []
    for word in s.split():
        if "'" in word:
            word = "'".join(fix(p) for p in word.split("'"))
        else:
            word = fix(word)
        out.append(word)
    return " ".join(out)


def split_name(raw: str | None) -> tuple[str | None, str | None]:
    """Split ``"DOE, JANE"`` or ``"Jane Doe"`` into ``(given, family)``."""
    if not raw or not str(raw).strip():
        return None, None
    s = " ".join(str(raw).split())
    if "," in s:
        family, given = (x.strip() for x in s.split(",", 1))
    else:
        parts = s.split(" ")
        given, family = (" ".join(parts[:-1]), parts[-1]) if len(parts) > 1 else (None, parts[0])
    return (smart_title(given) if given else None), (smart_title(family) if family else None)


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SOFT_BREAK = re.compile(r"(?<![.:\n])\n(?!\n|\d+\.|[A-Z][A-Za-z &/-]{2,40}:)")


def clean_text(raw: str) -> str:
    """Undo common scanner/OCR artifacts while preserving section structure.

    - re-joins words hyphenated across line breaks
    - joins soft-wrapped lines (but keeps lines that start a section or list item)
    - collapses runs of spaces and strips form feeds / trailing whitespace
    """
    s = raw.replace("\r\n", "\n").replace("\f", "\n")
    s = _HYPHEN_BREAK.sub(r"\1\2", s)
    s = _SOFT_BREAK.sub(" ", s)
    lines = [" ".join(line.split()) for line in s.split("\n")]
    s = "\n".join(lines)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def fingerprint(*parts: object) -> str:
    """Content hash that ignores whitespace and case, used to find duplicate documents."""
    norm = "|".join(" ".join(str(p).lower().split()) for p in parts)
    return hashlib.sha256(norm.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Lab values
# ---------------------------------------------------------------------------


def parse_float(raw: object) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        return float(s.lstrip("<>="))
    except ValueError:
        return None


def parse_ref_range(ref: str | None) -> tuple[float | None, float | None]:
    if not ref:
        return None, None
    s = ref.strip()
    if m := re.match(r"^([\d.]+)\s*-\s*([\d.]+)$", s):
        return float(m[1]), float(m[2])
    if m := re.match(r"^<\s*([\d.]+)$", s):
        return None, float(m[1])
    if m := re.match(r"^>\s*([\d.]+)$", s):
        return float(m[1]), None
    return None, None


def derive_flag(value: float | None, ref: str | None, given: str | None) -> tuple[str | None, str | None]:
    """Return (flag, note). Trusts the source flag but recomputes it from the range when they disagree."""
    from .labs import normalize_flag  # local import avoids a cycle at module load

    given_flag = normalize_flag(given)
    if value is None:
        return given_flag, None
    lo, hi = parse_ref_range(ref)
    computed = "N"
    if hi is not None and (value > hi or (ref or "").startswith("<") and value >= hi):
        computed = "H"
    elif lo is not None and value < lo:
        computed = "L"
    if lo is None and hi is None:
        return given_flag or None, None
    if given_flag is None:
        return computed, ("flag derived from reference range" if computed != "N" else None)
    # Critical flags (HH/LL) carry information the range can't: keep them when the direction agrees.
    if given_flag in ("HH", "LL") and given_flag[0] == computed:
        return given_flag, None
    if given_flag == "A" and computed in ("H", "L"):
        return computed, f"source flag 'A' (abnormal) refined to {computed!r} from range {ref!r}"
    if given_flag != computed:
        return computed, f"source flag {given_flag!r} disagreed with range {ref!r}; recomputed as {computed!r}"
    return given_flag, None
