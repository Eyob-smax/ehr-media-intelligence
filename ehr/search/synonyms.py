"""Small lay-term -> clinical-term expansion table.

General-purpose sentence embeddings (MiniLM) don't reliably connect everyday
phrasing to chart vocabulary: "blood clot" vs "thrombus", "kidney stone" vs
"ureteral calculus". A short curated table closes the most common gaps and is
easy to audit. A production system would use a clinical embedding model or a
UMLS/SNOMED synonym service instead (see README "What I'd improve").
"""

from __future__ import annotations

import re

EXPANSIONS: dict[str, list[str]] = {
    "blood clot": ["thrombus", "thrombosis", "dvt", "embolism"],
    "clot": ["thrombus", "thrombosis"],
    # Avoid findings that are usually documented as negatives ("no hydronephrosis") as synonyms.
    "kidney stone": ["calculus", "nephrolithiasis", "ureteral", "ureterovesical"],
    "heart failure": ["chf", "cardiomegaly", "ejection fraction", "lvef"],
    "fluid overload": ["edema", "effusion", "congestion"],
    "heart attack": ["myocardial infarction", "stemi"],
    "stroke": ["infarct", "tia", "ischemic attack"],
    "high blood pressure": ["hypertension"],
    "high blood sugar": ["hyperglycemia", "a1c", "glucose"],
    "sugar": ["glucose", "a1c"],
    "lung infection": ["pneumonia", "consolidation"],
    "chest x-ray": ["xr chest", "radiograph"],
    "x-ray": ["xr", "radiograph"],
    "thyroid": ["tsh", "hypothyroidism"],
    "back pain": ["lumbar", "radiculopathy", "disc"],
    "headache": ["migraine"],
    "kidney disease": ["ckd", "renal", "creatinine"],
    "kidney function": ["creatinine", "renal"],
    "high potassium": ["hyperkalemia", "potassium high"],
    "low blood count": ["anemia", "hemoglobin"],
    "breathing": ["dyspnea", "shortness of breath"],
    "short of breath": ["dyspnea"],
    "brain scan": ["ct head", "mri brain"],
    "urine infection": ["urinary tract infection", "uti", "urosepsis"],
}

_PATTERNS = [(re.compile(rf"\b{re.escape(k)}\b", re.IGNORECASE), k) for k in
             sorted(EXPANSIONS, key=len, reverse=True)]


STOPWORDS = set("a an and are as at be by for from has have in is it of on or the to was were with what which who "
                "show find patients patient records record any all me".split())


def expand(query: str) -> tuple[str, list[list[str]]]:
    """Return (embedding text, lexical phrase groups).

    Each group lists interchangeable phrases; a document satisfies the group if
    it contains any of them as whole words. Query words not covered by a known
    phrase become single-phrase groups (stopwords dropped).
    """
    remaining = query
    groups: list[list[str]] = []
    extra: list[str] = []
    for pat, key in _PATTERNS:
        if pat.search(remaining):
            remaining = pat.sub(" ", remaining)
            extra += EXPANSIONS[key]
            groups.append([key, *EXPANSIONS[key]])
    words = [w for w in re.findall(r"[a-z0-9][a-z0-9-]*", remaining.lower()) if w not in STOPWORDS and len(w) > 1]
    embed_text = f"{query} ({', '.join(extra)})" if extra else query
    return embed_text, groups + [[w] for w in words]


def group_matches(group: list[str], text_lower: str) -> bool:
    return any(re.search(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", text_lower) for p in group)
