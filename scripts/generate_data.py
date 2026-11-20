"""Generate a deterministic, deliberately messy synthetic EHR dataset.

Everything here is fictional. No real patient data is used or derived.

Outputs (under data/raw/ by default):
    ehr_export.json      Simulated EHR export: encounters with embedded notes and imaging reports
    lab_results.csv      One row per analyte, the way many LIS extracts look
    notes/*.txt          "Scanned" documents with OCR-style noise and free-text headers

Data quality problems are injected on purpose so the cleaning pipeline has
real work to do: inconsistent key names, five date formats, two-digit years,
several gender encodings, five MRN styles, missing fields, exact and
near-duplicate records, an MRN typo across sources, and a day/month-swapped DOB.

Usage:
    python scripts/generate_data.py [--out data/raw] [--seed 7]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Clinical building blocks
# ---------------------------------------------------------------------------

# (panel, test, unit, ref_low, ref_high)
ANALYTES = {
    "hgb": ("CBC", "Hemoglobin", "g/dL", 12.0, 17.5),
    "wbc": ("CBC", "WBC", "10*3/uL", 4.0, 11.0),
    "plt": ("CBC", "Platelets", "10*3/uL", 150, 400),
    "na": ("BMP", "Sodium", "mmol/L", 135, 145),
    "k": ("BMP", "Potassium", "mmol/L", 3.5, 5.1),
    "cr": ("BMP", "Creatinine", "mg/dL", 0.6, 1.3),
    "glu": ("BMP", "Glucose", "mg/dL", 70, 99),
    "a1c": ("HbA1c", "Hemoglobin A1c", "%", 4.0, 5.6),
    "ldl": ("Lipid Panel", "LDL Cholesterol", "mg/dL", 0, 99),
    "hdl": ("Lipid Panel", "HDL Cholesterol", "mg/dL", 40, 100),
    "chol": ("Lipid Panel", "Total Cholesterol", "mg/dL", 0, 199),
    "tsh": ("TSH", "TSH", "mIU/L", 0.4, 4.0),
    "bnp": ("BNP", "BNP", "pg/mL", 0, 100),
}

NORMAL_VALUES = {
    "hgb": 14.1, "wbc": 7.2, "plt": 255, "na": 139, "k": 4.2, "cr": 0.9,
    "glu": 92, "a1c": 5.3, "ldl": 88, "hdl": 52, "chol": 172, "tsh": 1.9, "bnp": 45,
}

PANELS = {
    "CBC": ["hgb", "wbc", "plt"],
    "BMP": ["na", "k", "cr", "glu"],
    "HbA1c": ["a1c"],
    "Lipid Panel": ["ldl", "hdl", "chol"],
    "TSH": ["tsh"],
    "BNP": ["bnp"],
}


@dataclass
class Condition:
    key: str
    reason: str
    setting: str  # inpatient | outpatient | emergency
    diagnoses: list[str]
    hpi: str
    course: str
    plan: str
    imaging: tuple[str, str, str] | None = None  # (study title, findings, impression)
    labs: dict[str, dict[str, float]] = field(default_factory=dict)  # panel -> overrides


CONDITIONS: list[Condition] = [
    Condition(
        "chf", "Shortness of breath and bilateral leg swelling", "inpatient",
        ["Acute on chronic systolic heart failure", "Essential hypertension"],
        "Progressive exertional dyspnea over 10 days, 3-pillow orthopnea and a 4 kg weight gain. "
        "Reports running out of furosemide two weeks ago.",
        "Diuresed with IV furosemide with net negative 5.2 L. Transitioned to oral diuretic. "
        "Echocardiogram showed LVEF 30%.",
        "Furosemide 40 mg PO daily, daily weights, low-sodium diet, cardiology follow-up in 1 week.",
        ("XR Chest 2 Views",
         "Enlarged cardiac silhouette. Bilateral small pleural effusions. Pulmonary vascular congestion.",
         "Cardiomegaly with pulmonary edema and bilateral pleural effusions."),
        {"BMP": {"na": 132, "k": 5.4, "cr": 1.6, "glu": 118}, "BNP": {"bnp": 1250}},
    ),
    Condition(
        "pna", "Fever and productive cough", "inpatient",
        ["Community-acquired pneumonia, right lower lobe"],
        "Four days of fever to 39.2 C, productive cough with rust-colored sputum and pleuritic right-sided chest pain.",
        "Treated with ceftriaxone and azithromycin. Defervesced by hospital day 2. Oxygen weaned to room air.",
        "Complete 5-day course of oral amoxicillin-clavulanate. Repeat chest x-ray in 6 weeks.",
        ("XR Chest PA and Lateral",
         "Dense airspace opacity in the right lower lobe with air bronchograms. No pneumothorax.",
         "Right lower lobe consolidation consistent with pneumonia."),
        {"CBC": {"wbc": 15.8, "hgb": 13.1, "plt": 310}},
    ),
    Condition(
        "dm2", "Follow-up for diabetes", "outpatient",
        ["Type 2 diabetes mellitus with hyperglycemia", "Obesity", "Hyperlipidemia"],
        "Here for routine diabetes follow-up. Reports inconsistent metformin use and increased thirst.",
        "Foot exam normal. Monofilament sensation intact. Retinal exam due.",
        "Increase metformin to 1000 mg BID, start atorvastatin 40 mg nightly, refer to diabetes education, repeat A1c in 3 months.",
        None,
        {"HbA1c": {"a1c": 9.4}, "BMP": {"glu": 212}, "Lipid Panel": {"ldl": 162, "chol": 248}},
    ),
    Condition(
        "ckd", "Nephrology follow-up for chronic kidney disease", "outpatient",
        ["Chronic kidney disease stage 3b", "Anemia of chronic kidney disease"],
        "Known CKD secondary to hypertensive nephrosclerosis. Reports mild fatigue. No edema.",
        "Creatinine above prior baseline of 1.8. Potassium elevated.",
        "Low-potassium diet, hold lisinopril pending repeat BMP in 1 week, start ESA evaluation.",
        ("US Renal Bilateral",
         "Right kidney 9.1 cm, left kidney 9.4 cm. Increased cortical echogenicity bilaterally. No hydronephrosis.",
         "Bilateral echogenic kidneys consistent with medical renal disease. No hydronephrosis."),
        {"BMP": {"cr": 2.1, "k": 5.6}, "CBC": {"hgb": 9.8}},
    ),
    Condition(
        "tia", "Sudden left arm weakness", "emergency",
        ["Transient ischemic attack", "Atrial fibrillation, new onset"],
        "Twenty minutes of left arm weakness and slurred speech that fully resolved before arrival.",
        "Neurologic exam normal on arrival. Telemetry showed atrial fibrillation with rates 90-110.",
        "Start apixaban 5 mg BID, atorvastatin 80 mg, outpatient echocardiogram and neurology follow-up.",
        ("CT Head without Contrast",
         "No acute intracranial hemorrhage, mass effect, or territorial infarct. Mild chronic small vessel ischemic change.",
         "No acute intracranial abnormality."),
        {"CBC": {}, "Lipid Panel": {"ldl": 141}},
    ),
    Condition(
        "appy", "Right lower quadrant abdominal pain", "inpatient",
        ["Acute appendicitis, status post laparoscopic appendectomy"],
        "Periumbilical pain migrating to the right lower quadrant over 18 hours with anorexia and nausea.",
        "Underwent uncomplicated laparoscopic appendectomy. Tolerated diet on postoperative day 1.",
        "No lifting over 10 lb for 2 weeks. Surgical clinic follow-up in 10 days.",
        ("CT Abdomen and Pelvis with Contrast",
         "Dilated appendix measuring 11 mm with periappendiceal fat stranding. No abscess or free air.",
         "Acute appendicitis without perforation."),
        {"CBC": {"wbc": 14.2}},
    ),
    Condition(
        "copd", "Worsening shortness of breath", "inpatient",
        ["COPD with acute exacerbation", "Tobacco use disorder"],
        "Increased dyspnea, wheeze and sputum production for 3 days. 40 pack-year smoking history.",
        "Treated with nebulized bronchodilators, prednisone 40 mg and doxycycline. Ambulatory saturation 91% on room air at discharge.",
        "Prednisone taper, tiotropium daily, smoking cessation counseling, pulmonology follow-up.",
        ("XR Chest 2 Views",
         "Hyperinflated lungs with flattened hemidiaphragms. No focal consolidation or effusion.",
         "Hyperinflation consistent with COPD. No acute process."),
        {"CBC": {"wbc": 11.9}},
    ),
    Condition(
        "hypothyroid", "Fatigue and weight gain", "outpatient",
        ["Primary hypothyroidism"],
        "Six months of fatigue, cold intolerance, constipation and 5 kg weight gain.",
        "Thyroid not enlarged on exam. Delayed relaxation of reflexes.",
        "Start levothyroxine 50 mcg daily, repeat TSH in 6 weeks.",
        None,
        {"TSH": {"tsh": 11.2}, "Lipid Panel": {"ldl": 128}},
    ),
    Condition(
        "dvt", "Left calf pain and swelling", "emergency",
        ["Deep vein thrombosis, left popliteal vein"],
        "Three days of left calf pain and swelling after a 9-hour flight. No chest pain or dyspnea.",
        "Left calf circumference 3 cm greater than right. Wells score 3.",
        "Start rivaroxaban 15 mg BID for 21 days then 20 mg daily. Return precautions for chest pain or dyspnea.",
        ("US Venous Duplex Left Lower Extremity",
         "Non-compressible left popliteal and posterior tibial veins with occlusive echogenic thrombus.",
         "Acute occlusive deep vein thrombosis of the left popliteal and posterior tibial veins."),
        {"CBC": {}},
    ),
    Condition(
        "stone", "Right flank pain", "emergency",
        ["Right ureteral calculus with hydronephrosis"],
        "Sudden severe colicky right flank pain radiating to the groin with nausea and gross hematuria.",
        "Pain controlled with ketorolac. Tolerating oral intake.",
        "Tamsulosin 0.4 mg daily, strain urine, urology follow-up within 1 week, return if fever.",
        ("CT Abdomen and Pelvis without Contrast",
         "6 mm calculus at the right ureterovesical junction with mild right hydroureteronephrosis.",
         "Obstructing 6 mm right UVJ calculus with mild hydronephrosis."),
        {"BMP": {"cr": 1.3}},
    ),
    Condition(
        "migraine", "Recurrent headaches", "outpatient",
        ["Migraine without aura"],
        "Throbbing unilateral headaches 3-4 times per month with photophobia and nausea. No red-flag features.",
        "Neurologic exam normal. MRI obtained given change in headache pattern.",
        "Sumatriptan 50 mg as needed, headache diary, consider prophylaxis if frequency increases.",
        ("MRI Brain without Contrast",
         "Normal brain parenchyma. No mass, hemorrhage or abnormal enhancement.",
         "Normal MRI of the brain."),
        {},
    ),
    Condition(
        "lbp", "Low back pain radiating to left leg", "outpatient",
        ["Lumbar disc herniation L4-L5 with left radiculopathy"],
        "Eight weeks of low back pain radiating down the posterior left leg to the foot. Failed 6 weeks of physical therapy.",
        "Positive straight leg raise on the left. Strength 5/5. No bowel or bladder symptoms.",
        "Refer to spine clinic for epidural steroid injection evaluation, continue physical therapy.",
        ("MRI Lumbar Spine without Contrast",
         "Left paracentral disc extrusion at L4-L5 with left lateral recess stenosis contacting the descending L5 nerve root.",
         "L4-L5 left paracentral disc extrusion with left lateral recess stenosis."),
        {},
    ),
    Condition(
        "ida", "Fatigue and lightheadedness", "outpatient",
        ["Iron deficiency anemia"],
        "Two months of fatigue, exertional lightheadedness and heavy menstrual bleeding.",
        "Conjunctival pallor. Ferritin 6 ng/mL.",
        "Ferrous sulfate 325 mg every other day, gynecology referral, repeat CBC in 4 weeks.",
        None,
        {"CBC": {"hgb": 7.9, "plt": 452}},
    ),
    Condition(
        "htn", "Hypertension follow-up", "outpatient",
        ["Essential hypertension", "Hyperlipidemia"],
        "Home blood pressures averaging 152/94. Adherent to amlodipine. No headache or chest pain.",
        "Office blood pressure 156/96.",
        "Add chlorthalidone 12.5 mg daily, repeat BMP in 2 weeks, continue home monitoring.",
        None,
        {"BMP": {}, "Lipid Panel": {"ldl": 131}},
    ),
    Condition(
        "urosepsis", "Fever and confusion", "inpatient",
        ["Sepsis due to urinary tract infection", "Acute kidney injury"],
        "Brought in by family for 1 day of confusion, fever to 38.9 C and dysuria.",
        "Blood and urine cultures grew E. coli. Treated with IV ceftriaxone and fluids. Mental status returned to baseline.",
        "Oral cephalexin to complete 10 days, repeat BMP in 1 week, primary care follow-up.",
        None,
        {"CBC": {"wbc": 18.9}, "BMP": {"cr": 1.8, "na": 133}},
    ),
    Condition(
        "asthma", "Wheezing and chest tightness", "emergency",
        ["Moderate persistent asthma with acute exacerbation"],
        "Two days of wheezing after a viral illness. Using albuterol every 3 hours.",
        "Improved after three albuterol-ipratropium nebulizers and oral dexamethasone. Peak flow 78% predicted at discharge.",
        "Start budesonide-formoterol, albuterol as needed, asthma action plan reviewed.",
        None,
        {"CBC": {}},
    ),
]


@dataclass
class Person:
    mrn: int
    given: str
    family: str
    dob: date
    gender: str  # male | female
    condition: Condition
    prior: Condition | None = None  # older encounter, to give patients a history


PEOPLE_SPEC = [
    ("Marisol", "Quintero", "1958-02-14", "female", "chf", "htn"),
    ("Theodore", "Banks", "1949-11-03", "male", "pna", "copd"),
    ("Aaliyah", "Okafor", "1972-06-21", "female", "dm2", None),
    ("Gregor", "Lindqvist", "1955-09-30", "male", "ckd", "htn"),
    ("Priya", "Raman", "1961-07-04", "female", "tia", None),
    ("Jonah", "Feld", "1994-12-09", "male", "appy", None),
    ("Delphine", "Moreau", "1947-03-17", "female", "copd", None),
    ("Samuel", "Achterberg", "1983-08-25", "male", "hypothyroid", None),
    ("Keiko", "Watanabe", "1979-01-12", "female", "dvt", None),
    ("Ruben", "Castellanos", "1988-05-02", "male", "stone", None),
    ("Imogen", "Thackeray", "1991-10-19", "female", "migraine", None),
    ("Darnell", "Whitfield", "1976-04-28", "male", "lbp", None),
    ("Fatima", "Benali", "1986-02-07", "female", "ida", None),
    ("Oskar", "Brandt", "1966-12-31", "male", "htn", "dm2"),
    ("Lucinda", "Harrow", "1940-08-08", "female", "urosepsis", "ckd"),
    ("Mateo", "Ibarra", "2001-03-15", "male", "asthma", None),
]

ANCHOR = date(2025, 10, 1)  # "today" for the synthetic dataset

# ---------------------------------------------------------------------------
# Mess generators
# ---------------------------------------------------------------------------


def mrn_variant(rng: random.Random, n: int) -> str:
    style = rng.choice(["canon", "bare", "padded", "lower", "spaced"])
    return {
        "canon": f"MRN-{n:07d}",
        "bare": str(n),
        "padded": f"{n:07d}",
        "lower": f"mrn{n}",
        "spaced": f"MRN {n:07d}",
    }[style]


def date_variant(rng: random.Random, d: date, allow_two_digit: bool = False) -> str:
    styles = ["iso", "us", "long", "slash_iso", "dmy_text"]
    if allow_two_digit:
        styles.append("us2")
    style = rng.choice(styles)
    return {
        "iso": d.isoformat(),
        "us": f"{d.month:02d}/{d.day:02d}/{d.year}",
        "long": d.strftime("%B %d, %Y").replace(" 0", " "),
        "slash_iso": f"{d.year}/{d.month}/{d.day}",
        "dmy_text": d.strftime("%d %b %Y"),
        "us2": f"{d.month}/{d.day}/{d.strftime('%y')}",
    }[style]


def datetime_variant(rng: random.Random, dt: datetime) -> str:
    style = rng.choice(["iso", "iso_z", "us_time", "date_only"])
    return {
        "iso": dt.strftime("%Y-%m-%dT%H:%M:%S"),
        "iso_z": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "us_time": dt.strftime("%m/%d/%Y %H:%M"),
        "date_only": dt.strftime("%Y-%m-%d"),
    }[style]


def gender_variant(rng: random.Random, g: str) -> str:
    options = {
        "male": ["M", "male", "Male", "1", "MALE"],
        "female": ["F", "female", "Female", "2", "FEMALE"],
    }[g]
    return rng.choice(options)


def ocr_noise(rng: random.Random, text: str) -> str:
    """Simulate scanner artifacts: hard wraps, hyphenation, doubled spaces."""
    words = text.split(" ")
    out, line_len = [], 0
    for w in words:
        if line_len > 60 and rng.random() < 0.6:
            if len(w) > 7 and rng.random() < 0.3:
                cut = len(w) // 2
                out.append(w[:cut] + "-\n" + w[cut:])
            else:
                out.append("\n" + w)
            line_len = len(w)
        else:
            out.append(("  " if rng.random() < 0.08 else "") + w)
            line_len += len(w) + 1
    return " ".join(out)


# ---------------------------------------------------------------------------
# Document text
# ---------------------------------------------------------------------------


def note_text(p: Person, c: Condition, when: date, kind: str) -> str:
    dx = "\n".join(f"{i}. {d}" for i, d in enumerate(c.diagnoses, 1))
    age = (when - p.dob).days // 365
    sex = "woman" if p.gender == "female" else "man"
    if kind == "discharge_summary":
        return (
            f"DISCHARGE SUMMARY\n"
            f"Chief Complaint: {c.reason}.\n"
            f"History of Present Illness: {age}-year-old {sex} presenting with {c.reason.lower()}. {c.hpi}\n"
            f"Hospital Course: {c.course}\n"
            f"Discharge Diagnoses:\n{dx}\n"
            f"Discharge Plan: {c.plan}"
        )
    if kind == "ed_note":
        return (
            f"EMERGENCY DEPARTMENT NOTE\n"
            f"Chief Complaint: {c.reason}.\n"
            f"HPI: {age}-year-old {sex} with {c.reason.lower()}. {c.hpi}\n"
            f"ED Course: {c.course}\n"
            f"Assessment:\n{dx}\n"
            f"Disposition and Plan: {c.plan}"
        )
    return (
        f"PROGRESS NOTE\n"
        f"Reason for Visit: {c.reason}.\n"
        f"Subjective: {age}-year-old {sex}. {c.hpi}\n"
        f"Objective: {c.course}\n"
        f"Assessment:\n{dx}\n"
        f"Plan: {c.plan}"
    )


def imaging_text(c: Condition) -> str:
    study, findings, impression = c.imaging  # type: ignore[misc]
    return (
        f"{study.upper()}\n"
        f"Clinical Indication: {c.reason}.\n"
        f"Technique: Standard protocol.\n"
        f"Findings: {findings}\n"
        f"Impression: {impression}"
    )


def lab_rows(p: Person, c: Condition, collected: datetime) -> list[dict]:
    rows = []
    for panel, overrides in c.labs.items():
        for key in PANELS[panel]:
            _, test, unit, lo, hi = ANALYTES[key]
            value = overrides.get(key, NORMAL_VALUES[key])
            flag = "H" if value > hi else "L" if value < lo else ""
            ref = f"{lo}-{hi}" if lo else f"<{hi + (1 if isinstance(hi, int) else 0)}"
            rows.append({
                "panel": panel, "test": test, "value": value,
                "unit": unit, "ref_range": ref, "flag": flag,
                "collected": collected,
            })
    return rows


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------


def build(out: Path, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    cond_by_key = {c.key: c for c in CONDITIONS}

    people: list[Person] = []
    for i, (given, family, dob, gender, ck, prior) in enumerate(PEOPLE_SPEC):
        people.append(Person(
            mrn=1000 + i * 373 + rng.randint(0, 50),
            given=given, family=family, dob=date.fromisoformat(dob), gender=gender,
            condition=cond_by_key[ck], prior=cond_by_key[prior] if prior else None,
        ))

    json_records: list[dict] = []
    csv_rows: list[dict] = []
    txt_docs: list[tuple[str, str]] = []
    note_dates: dict[tuple[str, str], datetime] = {}
    rec_no = 0

    def patient_block(p: Person) -> dict:
        """Patient demographics with randomized key names and formats."""
        block: dict = {}
        block[rng.choice(["mrn", "MRN", "medical_record_number"])] = mrn_variant(rng, p.mrn)
        if rng.random() < 0.5:
            block["name"] = f"{p.family.upper()}, {p.given}" if rng.random() < 0.5 else f"{p.given} {p.family}"
        else:
            block["first_name"], block["last_name"] = p.given, p.family
        block[rng.choice(["dob", "DOB", "birthDate", "date_of_birth"])] = date_variant(rng, p.dob)
        block[rng.choice(["gender", "sex"])] = gender_variant(rng, p.gender)
        return block

    def add_encounter(p: Person, c: Condition, when: datetime) -> None:
        nonlocal rec_no
        rec_no += 1
        enc_id = f"ENC-{rec_no:05d}"
        docs = []
        note_kind = {"inpatient": "discharge_summary", "emergency": "ed_note", "outpatient": "progress_note"}[c.setting]
        note_date = when + timedelta(days=3 if c.setting == "inpatient" else 0)
        note_dates[(p.given, c.key)] = note_date
        docs.append({
            "doc_type": note_kind,
            "title": {"discharge_summary": "Discharge Summary", "ed_note": "ED Provider Note",
                      "progress_note": "Office Visit Note"}[note_kind],
            rng.choice(["date", "created", "authored"]): datetime_variant(rng, note_date),
            "text": note_text(p, c, note_date.date(), note_kind),
        })
        if c.imaging:
            img_dt = when + timedelta(hours=rng.randint(1, 6))
            docs.append({
                "doc_type": "imaging_report",
                "title": c.imaging[0],
                rng.choice(["date", "performed"]): datetime_variant(rng, img_dt),
                "text": imaging_text(c),
            })
        json_records.append({
            "record_id": f"REC-{rec_no:05d}",
            "source_system": "EpicSim",
            "patient": patient_block(p),
            "encounter": {
                "id": enc_id,
                rng.choice(["date", "admit_date", "start"]): datetime_variant(rng, when),
                "class": c.setting,
                "reason": c.reason,
            },
            "documents": docs,
        })
        for row in lab_rows(p, c, when + timedelta(hours=1)):
            csv_rows.append({"_person": p, "_encounter": enc_id, **row})

    # --- core encounters ---------------------------------------------------
    for idx, p in enumerate(people):
        if p.prior:
            prior_when = datetime.combine(ANCHOR - timedelta(days=rng.randint(200, 420)), datetime.min.time()) + timedelta(hours=9)
            add_encounter(p, p.prior, prior_when)
        when = datetime.combine(ANCHOR - timedelta(days=rng.randint(3, 150)), datetime.min.time()) + timedelta(hours=rng.randint(7, 20))
        add_encounter(p, p.condition, when)

    # --- injected data-quality problems -----------------------------------
    # 1. Exact duplicate of an export record (re-sent message).
    json_records.append(json.loads(json.dumps(json_records[2])))
    # 2. Near-duplicate: same note, different whitespace/case in the text, new record id.
    near = json.loads(json.dumps(json_records[5]))
    near["record_id"] = "REC-90001"
    near["documents"][0]["text"] = "  " + near["documents"][0]["text"].replace(". ", ".  ") + "\n"
    json_records.append(near)
    # 3. Missing fields: drop gender for one patient, DOB for another, a doc date for a third.
    for rec in json_records:
        if rec["encounter"]["reason"] == "Recurrent headaches":
            rec["patient"].pop("gender", None); rec["patient"].pop("sex", None)
        if rec["encounter"]["reason"] == "Wheezing and chest tightness":
            for k in ("dob", "DOB", "birthDate", "date_of_birth"):
                rec["patient"].pop(k, None)
        if rec["encounter"]["reason"] == "Right flank pain":
            for k in ("date", "created", "authored"):
                rec["documents"][0].pop(k, None)
    # 4. Day/month swapped DOB for Priya Raman in a second record (1961-07-04 -> 1961-04-07).
    priya = next(p for p in people if p.given == "Priya")
    rec_no += 1
    json_records.append({
        "record_id": f"REC-{rec_no:05d}",
        "source_system": "EpicSim",
        "patient": {"MRN": f"MRN-{priya.mrn:07d}", "name": "Priya Raman", "dob": "1961-04-07", "sex": "F"},
        "encounter": {"id": f"ENC-{rec_no:05d}", "date": (ANCHOR - timedelta(days=1)).isoformat(),
                      "class": "outpatient", "reason": "Post-TIA follow-up"},
        "documents": [{
            "doc_type": "progress_note", "title": "Neurology Follow-up",
            "date": (ANCHOR - timedelta(days=1)).isoformat(),
            "text": ("PROGRESS NOTE\nReason for Visit: Post-TIA follow-up.\n"
                     "Subjective: No recurrent focal symptoms. Tolerating apixaban without bleeding.\n"
                     "Objective: Neurologic exam normal. Echocardiogram showed mildly dilated left atrium, LVEF 60%.\n"
                     "Assessment:\n1. Transient ischemic attack, resolved\n2. Atrial fibrillation on anticoagulation\n"
                     "Plan: Continue apixaban and atorvastatin. Recheck lipid panel in 3 months."),
        }],
    })

    # --- CSV lab rows ------------------------------------------------------
    gregor = next(p for p in people if p.given == "Gregor")
    csv_out: list[dict] = []
    for r in csv_rows:
        p: Person = r["_person"]
        mrn_str = mrn_variant(rng, p.mrn)
        # 5. MRN typo for Gregor: two digits transposed in the lab system.
        if p is gregor and r["panel"] == "CBC":
            s = f"{p.mrn:07d}"
            mrn_str = s[:-2] + s[-1] + s[-2]
        name = rng.choice([f"{p.family.upper()}, {p.given.upper()}", f"{p.given} {p.family}", f"{p.family}, {p.given}"])
        csv_out.append({
            "mrn": mrn_str,
            "patient_name": name,
            "dob": date_variant(rng, p.dob, allow_two_digit=True),
            "collected": rng.choice([
                r["collected"].strftime("%m/%d/%Y %H:%M"),
                r["collected"].strftime("%Y-%m-%d %H:%M"),
                r["collected"].strftime("%d-%b-%Y"),
            ]),
            "panel": r["panel"], "test": r["test"], "value": r["value"],
            "unit": r["unit"], "ref_range": r["ref_range"], "flag": r["flag"],
        })
    # 6. Duplicate lab rows (interface resend) and a row with a missing value.
    csv_out.extend(dict(row) for row in csv_out[4:7])
    csv_out.insert(10, {**csv_out[10], "value": "", "flag": ""})
    # 7. Missing patient_name on a few rows (must be back-filled from MRN).
    for row in csv_out[20:23]:
        row["patient_name"] = ""

    # --- Scanned text documents --------------------------------------------
    def scanned(p: Person, doc_type: str, when: date, body: str) -> str:
        header_name = rng.choice([f"{p.family.upper()}, {p.given.upper()}", f"{p.given} {p.family}"])
        header = (
            f"PATIENT: {header_name}    MRN: {mrn_variant(rng, p.mrn)}\n"
            f"DOB: {date_variant(rng, p.dob, allow_two_digit=True)}    SEX: {gender_variant(rng, p.gender)}\n"
            f"DATE OF SERVICE: {date_variant(rng, when)}\n"
            f"DOCUMENT TYPE: {doc_type}\n"
            f"{'-' * 40}\n"
        )
        return header + ocr_noise(rng, body) + "\n\f"

    outside = [
        ("Theodore", "Outside Radiology Report", 30,
         "XR CHEST PA AND LATERAL (OUTSIDE FACILITY)\nFindings: Interval improvement of right lower lobe opacity with residual "
         "patchy atelectasis. No new consolidation.\nImpression: Improving right lower lobe pneumonia."),
        ("Marisol", "Cardiology Consult Letter", 10,
         "Dear colleague, I saw Ms. Quintero for heart failure with reduced ejection fraction, LVEF 30 percent. "
         "She remains NYHA class III. I have started sacubitril-valsartan and metoprolol succinate. "
         "Please monitor potassium and creatinine within one week."),
        ("Keiko", "Outside Records - Urgent Care", 5,
         "Urgent care visit for left leg swelling. D-dimer elevated at 1.9 mg/L FEU. Referred to emergency department "
         "for duplex ultrasound to rule out DVT."),
        ("Lucinda", "Skilled Nursing Facility Transfer Note", 2,
         "Transferred to skilled nursing for rehabilitation after urosepsis. Baseline mild dementia. Ambulates with walker. "
         "Allergy: sulfa drugs. Fall risk high."),
        ("Darnell", "Physical Therapy Evaluation", 40,
         "Physical therapy evaluation for lumbar radiculopathy. Limited lumbar flexion. Pain 7/10. "
         "Plan for core stabilization twice weekly for 6 weeks."),
        ("Aaliyah", "Ophthalmology Report", 25,
         "Dilated retinal exam: mild nonproliferative diabetic retinopathy in both eyes without macular edema. "
         "Recommend annual follow-up and improved glycemic control."),
    ]
    for given, doc_type, days_ago, body in outside:
        p = next(x for x in people if x.given == given)
        when = ANCHOR - timedelta(days=days_ago)
        slug = f"{p.family.lower()}_{doc_type.split()[0].lower()}_{when:%Y%m%d}.txt"
        txt_docs.append((slug, scanned(p, doc_type, when, body)))

    # 8. A scanned copy of an export discharge summary (cross-format duplicate).
    jonah = next(p for p in people if p.given == "Jonah")
    jonah_rec = next(r for r in json_records if r["encounter"]["reason"] == "Right lower quadrant abdominal pain")
    jonah_date = note_dates[(jonah.given, "appy")]
    txt_docs.append((f"feld_discharge_scan_{jonah_date:%Y%m%d}.txt",
                     scanned(jonah, "Discharge Summary", jonah_date.date(), jonah_rec["documents"][0]["text"])))

    # --- write -------------------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    (out / "notes").mkdir(exist_ok=True)
    for f in (out / "notes").glob("*.txt"):
        f.unlink()
    rng.shuffle(json_records)
    (out / "ehr_export.json").write_text(json.dumps({"export_date": ANCHOR.isoformat(), "records": json_records}, indent=2))
    with (out / "lab_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_out[0].keys()))
        w.writeheader()
        w.writerows(csv_out)
    for name, body in txt_docs:
        (out / "notes" / name).write_text(body)

    return {"json_records": len(json_records), "lab_rows": len(csv_out), "scanned_docs": len(txt_docs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "raw")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    counts = build(args.out, args.seed)
    print(f"Wrote synthetic dataset to {args.out}: {counts}")


if __name__ == "__main__":
    main()
