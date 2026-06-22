"""
Rescore the v3 responses with a medically-aware stricter rubric.

Key changes vs the original rubric:
1. Mode 1 cannot win on safety-critical categories (drug interactions, food,
   lifestyle, side effects). Even a confident-sounding LLM answer is 0
   because hallucination risk is too high without grounded sources.
2. Mode 1 wins on RAG categories only if it cites a specific patient value
   (not just the drug name).
3. Mode 1 still wins on drug_general_info and drug_mechanism (truly general).

Outputs:
  patient_portal_ground_truth_v3_2000_STRICT.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "synthetic_patients"


def load_jsonl(p: Path) -> list[dict]:
    with p.open() as f:
        return [json.loads(l) for l in f if l.strip()]


# Categories where Mode 1 is unsafe — LLM-alone should not be trusted
SAFETY_CRITICAL_CATEGORIES = {
    "drug_interaction",
    "drug_food_interaction",
    "drug_lifestyle_safety",
    "drug_side_effects",
}

# Categories where Mode 1 is OK if it answers reasonably
GENERAL_CATEGORIES = {
    "drug_general_info",
    "drug_mechanism",
}

# Categories where Mode 1 needs patient-specific evidence
PATIENT_SPECIFIC_CATEGORIES = {
    "patient_indication",
    "patient_timing",
    "patient_vitals",
    "patient_lifestyle",
    "patient_monitoring",
    "patient_side_effects",
}


PUNT_RE = re.compile(
    r"i can'?t give (you )?medical advice|i'?m sorry,? but i can'?t|"
    r"i am not (a doctor|a medical|able to)|"
    r"please (talk to|consult|speak to|discuss with) (your )?doctor|"
    r"only your doctor can|i recommend (you )?(speak|consult|talk) (to|with) (your )?doctor",
    re.IGNORECASE,
)


def is_pure_punt(a: str) -> bool:
    if not a:
        return True
    if len(a) < 250 and PUNT_RE.search(a):
        useful = sum(1 for s in re.split(r"[.!?]+", a)
                     if len(s.strip()) >= 15 and not PUNT_RE.search(s))
        return useful <= 1
    return False


def cites_patient_specific(a: str, p: dict) -> bool:
    """True if the answer references a specific value from the patient record."""
    al = a.lower()
    # Vitals values
    for k, v in p.get("recent_vitals", {}).items():
        if k == "date":
            continue
        v_str = str(v).lower()
        if v_str and v_str in al:
            return True
        if isinstance(v, (int, float)) and (str(v) in a or f"{v:.1f}" in a):
            return True
    # Allergy substance
    for x in p.get("allergies", []):
        if x.get("substance", "").lower() in al:
            return True
    # PMH item
    for h in p.get("past_medical_history", []):
        first = " ".join(h.lower().split()[:3])
        if first and first in al:
            return True
    # Specific condition mention with diagnosis date
    for c in p.get("active_conditions", []):
        cn = c["condition"].lower()
        if cn in al:
            return True
    return False


def mentions_drug_interaction(a: str) -> bool:
    al = a.lower()
    return any(k in al for k in [
        # explicit interaction language
        "interact", "increase the levels", "decrease the levels",
        "bleeding risk", "cyp", "metabolism", "metabolized",
        "qt prolongation", "serotonin syndrome", "lactic acidosis",
        "hypoglycemi", "hyperkalemi", "potassium", "potentiate",
        "additive effect", "reduce effectiveness",
        "could interact", "may interact", "can interact",
        # implicit interaction phrasing
        "combined with", "in combination with", "together with",
        "while taking", "concurrent use", "concurrently",
        "alongside", "when taken with", "when used with",
        "co-administer", "concomitant",
        # food / lifestyle interaction phrasing
        "food can affect", "with food", "without food",
        "empty stomach", "absorption", "bioavailability",
        "alcohol can", "drinking alcohol", "alcohol may",
    ])


def gives_safety_advice(a: str) -> bool:
    al = a.lower()
    return any(re.search(p, al) for p in [
        r"\bavoid\b", r"don'?t (take|drink|consume|combine)",
        r"do not (take|combine)", r"safe to (take|combine)",
        r"can take .* with", r"should not", r"contraindicated",
        r"limit (your )?intake", r"\bcaution\b", r"may cause",
        r"is generally safe", r"best to wait", r"separate .* by",
        r"hours apart", r"could interact", r"may interact",
    ])


def is_grounded(rec: dict) -> bool:
    """True if the mode actually used external context (chunks or KG triples)."""
    return (rec.get("n_retrieved_chunks", 0) > 0
            or rec.get("n_medlineplus_chunks", 0) > 0
            or rec.get("n_kg_triples", 0) > 0)


def score_strict(rec: dict, p: dict, category: str, rationale: str) -> tuple[int, str]:
    a = rec["answer"]
    m = rec["mode"]

    if not a or len(a) < 30:
        return 0, "empty/short"
    if is_pure_punt(a):
        return 0, "punted"

    # ----- SAFETY-CRITICAL CATEGORIES -----
    # Mode 1 (LLM only) cannot be trusted here regardless of how good it sounds
    if category in SAFETY_CRITICAL_CATEGORIES:
        if m == 1:
            return 0, "Mode 1 not trustworthy on safety question (no grounded sources)"

        # Drug-interaction subcategory: stricter on Mode 2 (must mention
        # interaction explicitly), more lenient on Mode 3 (KG triples in
        # the prompt count as grounding even if the wording is implicit).
        is_interaction_question = category in {"drug_interaction", "drug_food_interaction"}
        if is_interaction_question:
            has_kg = rec.get("n_kg_triples", 0) > 0
            if m == 3 and has_kg and gives_safety_advice(a):
                return 1, "Mode 3: KG-grounded + safety advice"
            if mentions_drug_interaction(a) and gives_safety_advice(a):
                return 1, "interaction language + advice"
            if m == 3 and has_kg and len(a) >= 200:
                return 1, "Mode 3: KG-grounded informative answer"
            return 0, "interaction question needs explicit grounding"

        # Lifestyle / side-effect safety: still need substantive answer
        if mentions_drug_interaction(a) and gives_safety_advice(a):
            return 1, "interaction + concrete safety advice"
        if gives_safety_advice(a):
            return 1, "concrete safety advice"
        if is_grounded(rec) and len(a) >= 200:
            return 1, "grounded informative answer"
        return 0, "missed safety/interaction without grounding"

    # ----- PATIENT-SPECIFIC CATEGORIES -----
    if category in PATIENT_SPECIFIC_CATEGORIES:
        if m == 1:
            # Allowed only if Mode 1 cites a specific patient value
            # (came from the prescription block in the prompt)
            if cites_patient_specific(a, p):
                return 1, "cited specific patient value"
            # Special case: indication questions where the prescription notes
            # already contain the answer (e.g. 'For: Hypertension control')
            if category == "patient_indication":
                meds = [m["drug"].split()[0].lower() for m in p.get("medications", [])]
                if any(d in a.lower() for d in meds) and len(a) >= 80:
                    return 1, "indication answered from prescription block"
            return 0, "Mode 1 needed patient context but didn't use any"
        # Mode 2/3 with grounded retrieval + substantive
        if is_grounded(rec) and len(a) >= 80:
            return 1, "grounded patient context"
        return 0, "no grounded context"

    # ----- GENERAL CATEGORIES -----
    if category in GENERAL_CATEGORIES:
        # Any non-punt answer of reasonable length is OK for any mode
        if len(a) >= 80:
            return 1, "general info answered"
        return 0, "too short"

    # Unknown category — be conservative
    return 0, "unknown category"


def main() -> None:
    resp_path = DATA / "patient_portal_responses_v3_2000.jsonl"
    if not resp_path.exists():
        print(f"ERROR: {resp_path} not found")
        sys.exit(1)
    pat_path = DATA / "patients_v2.jsonl"
    if not pat_path.exists():
        print(f"ERROR: {pat_path} not found")
        sys.exit(1)

    responses = load_jsonl(resp_path)
    patients = {p["patient_id"]: p for p in load_jsonl(pat_path)}
    print(f"Rescoring {len(responses)} responses with strict medically-aware rubric...\n")

    out = []
    for r in responses:
        p = patients.get(r["patient_id"])
        if p is None:
            continue
        cat = r["category"]
        rt = r.get("rationale", "")

        s1, w1 = score_strict(r["mode_1"], p, cat, rt)
        s2, w2 = score_strict(r["mode_2"], p, cat, rt)
        s3, w3 = score_strict(r["mode_3"], p, cat, rt)
        winner = 1 if s1 else (2 if s2 else (3 if s3 else 0))
        out.append({
            "question_id": r["question_id"],
            "patient_id": r["patient_id"],
            "category": cat,
            "expected_mode": r["expected_mode"],
            "expected_mode_name": r["expected_mode_name"],
            "mode_1_correct": s1, "mode_1_rationale": w1,
            "mode_2_correct": s2, "mode_2_rationale": w2,
            "mode_3_correct": s3, "mode_3_rationale": w3,
            "winning_mode": winner,
        })

    out_path = DATA / "patient_portal_ground_truth_v3_2000_STRICT.jsonl"
    with out_path.open("w") as f:
        for g in out:
            f.write(json.dumps(g) + "\n")
    print(f"Saved: {out_path}")

    n = len(out)
    print(f"\n=== STRICT rubric results (n={n}) ===")
    print(f"Mode 1: {sum(g['mode_1_correct'] for g in out)/n*100:.1f}%")
    print(f"Mode 2: {sum(g['mode_2_correct'] for g in out)/n*100:.1f}%")
    print(f"Mode 3: {sum(g['mode_3_correct'] for g in out)/n*100:.1f}%")
    oracle = sum(1 for g in out
                 if g['mode_1_correct'] or g['mode_2_correct'] or g['mode_3_correct']) / n * 100
    print(f"Oracle: {oracle:.1f}%")
    print()
    print("Winning mode distribution:")
    for m, c in sorted(Counter(g['winning_mode'] for g in out).items()):
        label = {1: "Mode 1 (LLM only)", 2: "Mode 2 (RAG)",
                  3: "Mode 3 (RAG+KG)", 0: "All wrong"}[m]
        print(f"  {label}: {c}  ({c/n*100:.1f}%)")
    print()
    print("By-category Mode 1 acceptance rate (lower = stricter penalty):")
    cat_m1 = {}
    for g in out:
        cat_m1.setdefault(g["category"], []).append(g["mode_1_correct"])
    for cat, vals in sorted(cat_m1.items()):
        rate = sum(vals) / len(vals) * 100
        print(f"  {cat:30s}  {rate:5.1f}%  (n={len(vals)})")


if __name__ == "__main__":
    main()
