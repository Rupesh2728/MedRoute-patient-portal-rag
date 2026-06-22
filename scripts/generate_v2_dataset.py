#!/usr/bin/env python3
"""
Synthetic Patient Dataset Generator v2
Expands 50 patients + 250 train + 100 test questions to:
- 400 patients (50 existing + 350 new)
- 1600 training questions
- 400 test questions
Total: 2000 questions across 400 patients
"""

import json
import random
import os
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Any, Tuple

# Set seed for reproducibility
random.seed(42)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SYNTHETIC_PATIENTS_DIR = PROJECT_ROOT / "synthetic_patients"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Input files
PATIENTS_FILE = SYNTHETIC_PATIENTS_DIR / "patients.jsonl"
QUESTIONS_FILE = SYNTHETIC_PATIENTS_DIR / "questions.jsonl"
TEST_QUESTIONS_FILE = SYNTHETIC_PATIENTS_DIR / "test_questions.jsonl"

# Output files
PATIENTS_V2_FILE = SYNTHETIC_PATIENTS_DIR / "patients_v2.jsonl"
QUESTIONS_V2_FILE = SYNTHETIC_PATIENTS_DIR / "questions_v2.jsonl"
TEST_QUESTIONS_V2_FILE = SYNTHETIC_PATIENTS_DIR / "test_questions_v2.jsonl"

# ============================================================================
# Data pools and templates
# ============================================================================

FIRST_NAMES = {
    "Male": ["James", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Charles",
             "Thomas", "Christopher", "Daniel", "Matthew", "Anthony", "Marcus", "Donald",
             "Kenneth", "Steven", "Paul", "Andrew", "Joshua", "Kevin", "Brian", "George",
             "Edward", "Ronald", "Timothy", "Jason", "Jeffrey", "Ryan", "Jacob", "Gary",
             "Nicholas", "Eric", "Jonathan", "Stephen", "Larry", "Justin", "Scott", "Brandon",
             "Benjamin", "Samuel", "Frank", "Gregory", "Raymond", "Patrick", "Jack", "Dennis"],
    "Female": ["Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth", "Susan", "Jessica",
               "Sarah", "Karen", "Nancy", "Lisa", "Betty", "Margaret", "Sandra", "Ashley",
               "Kimberly", "Emily", "Donna", "Michelle", "Dorothy", "Carol", "Amanda", "Melissa",
               "Deborah", "Stephanie", "Rebecca", "Sharon", "Laura", "Cynthia", "Kathleen",
               "Amy", "Angela", "Shirley", "Anna", "Brenda", "Pamela", "Emma", "Nicole",
               "Helen", "Samantha", "Katherine", "Christine", "Debra", "Rachel", "Catherine",
               "Carolyn", "Janet", "Ruth", "Maria"],
}

LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
              "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
              "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
              "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Young",
              "Diaz", "Allen", "King", "Wright", "Lopez", "Hill", "Scott", "Green", "Adams",
              "Nelson", "Carter", "Roberts", "Guzman", "Phillips", "Evans", "Turner", "Jimenez",
              "Edwards", "Collins", "Reyes", "Morris", "Morales", "Murphy"]

# Condition-medication mappings
CONDITION_MEDICATIONS = {
    "Type 2 Diabetes Mellitus": [
        ("Metformin", "500-2000mg", "Twice daily with meals"),
        ("Empagliflozin", "10-25mg", "Once daily"),
        ("Linagliptin", "5mg", "Once daily"),
        ("Pioglitazone", "15-45mg", "Once daily"),
        ("Glimepiride", "1-4mg", "Once daily"),
    ],
    "Hypertension": [
        ("Lisinopril", "10-40mg", "Once daily"),
        ("Amlodipine", "5-10mg", "Once daily"),
        ("Hydrochlorothiazide", "12.5-25mg", "Once daily"),
        ("Atenolol", "25-100mg", "Once daily"),
        ("Losartan", "50-100mg", "Once daily"),
    ],
    "Coronary Artery Disease (post-MI)": [
        ("Aspirin", "81mg", "Once daily"),
        ("Atorvastatin", "40-80mg", "Once daily"),
        ("Metoprolol Succinate", "25-190mg", "Once daily"),
        ("Clopidogrel", "75mg", "Once daily"),
    ],
    "Hyperlipidemia": [
        ("Atorvastatin", "20-80mg", "Once daily"),
        ("Rosuvastatin", "10-40mg", "Once daily"),
        ("Simvastatin", "20-40mg", "Once daily"),
        ("Ezetimibe", "10mg", "Once daily"),
    ],
    "Major Depressive Disorder": [
        ("Sertraline", "50-200mg", "Once daily"),
        ("Fluoxetine", "20-80mg", "Once daily"),
        ("Paroxetine", "20-50mg", "Once daily"),
        ("Citalopram", "20-40mg", "Once daily"),
        ("Escitalopram", "10-20mg", "Once daily"),
    ],
    "Generalized Anxiety Disorder": [
        ("Sertraline", "50-200mg", "Once daily"),
        ("Paroxetine", "20-50mg", "Once daily"),
        ("Venlafaxine", "75-225mg", "Once daily"),
        ("Buspirone", "15-30mg", "Twice daily"),
    ],
    "Persistent Asthma (moderate)": [
        ("Fluticasone Propionate (inhaled)", "110-220 mcg", "Twice daily"),
        ("Budesonide (inhaled)", "180-720 mcg", "Twice daily"),
        ("Albuterol (inhaled)", "90 mcg", "As needed"),
        ("Salmeterol (inhaled)", "50 mcg", "Twice daily"),
    ],
    "Allergic Rhinitis": [
        ("Loratadine", "10mg", "Once daily"),
        ("Cetirizine", "10mg", "Once daily"),
        ("Fexofenadine", "180mg", "Once daily"),
        ("Intranasal fluticasone", "110 mcg", "Twice daily"),
    ],
    "Hashimoto's Thyroiditis (Hypothyroidism)": [
        ("Levothyroxine", "25-200 mcg", "Once daily on empty stomach"),
        ("Liothyronine", "25 mcg", "Once daily"),
    ],
    "Obesity (BMI > 30)": [
        ("Phentermine", "15-30mg", "Once daily"),
        ("Orlistat", "120mg", "Three times daily with meals"),
    ],
    "Chronic Obstructive Pulmonary Disease": [
        ("Albuterol (inhaled)", "90 mcg", "Every 4-6 hours"),
        ("Tiotropium (inhaled)", "18 mcg", "Once daily"),
        ("Ipratropium/Albuterol", "combined", "Four times daily"),
    ],
    "Atrial Fibrillation": [
        ("Warfarin", "2-10mg", "Once daily"),
        ("Apixaban", "5mg", "Twice daily"),
        ("Metoprolol Succinate", "25-190mg", "Once daily"),
        ("Diltiazem", "120-360mg", "Once daily"),
    ],
}

ALLERGIES_POOL = [
    {"substance": "Penicillin", "reaction": "Rash and hives"},
    {"substance": "Sulfa drugs", "reaction": "Severe rash"},
    {"substance": "NSAIDs", "reaction": "GI upset and angioedema"},
    {"substance": "ACE inhibitors", "reaction": "Persistent cough"},
    {"substance": "Statins", "reaction": "Muscle pain"},
    {"substance": "Pollen (seasonal)", "reaction": "Rhinitis, watery eyes"},
    {"substance": "Dust mites", "reaction": "Asthma exacerbations"},
    {"substance": "Shellfish", "reaction": "Anaphylaxis"},
    {"substance": "Tree nuts", "reaction": "Throat swelling"},
]

# ============================================================================
# Question templates by category
# ============================================================================

QUESTION_TEMPLATES = {
    "drug_general_info": [
        "What is {drug_name} and why is it prescribed?",
        "How do I take {drug_name} correctly?",
        "What should I do if I miss a dose of {drug_name}?",
        "What are the main benefits of {drug_name}?",
        "Is {drug_name} available as a generic?",
        "What if I run out of {drug_name} while traveling?",
        "How long does {drug_name} stay in my system?",
        "Can I take {drug_name} with other over-the-counter medications?",
        "What storage conditions does {drug_name} require?",
        "Should I take {drug_name} with or without food?",
    ],
    "drug_mechanism": [
        "How does {drug_name} work to treat my condition?",
        "What does {drug_name} do for my body?",
        "Can you explain the mechanism of action of {drug_name}?",
        "How quickly does {drug_name} start working?",
        "What chemical processes does {drug_name} affect?",
        "Why is {drug_name} effective for managing {condition}?",
        "How does {drug_name} help with my {condition}?",
        "What is the pharmacological basis of {drug_name}?",
        "How does {drug_name} improve my health?",
        "What happens in my body when I take {drug_name}?",
    ],
    "drug_interaction": [
        "Can I take ibuprofen for headaches while on {drug_name}?",
        "Is there an interaction between {drug_name} and other medications?",
        "What medications should I avoid while taking {drug_name}?",
        "Can I take {drug_name} with my other prescriptions?",
        "Are there any dangerous drug combinations I should know about?",
        "Does {drug_name} interact with supplements?",
        "Can {drug_name} be taken with blood thinners?",
        "What about taking {drug_name} with antacids?",
        "Does {drug_name} work well with vitamin supplements?",
        "Can I combine {drug_name} with other heart medications?",
    ],
    "drug_food_interaction": [
        "Should I eat before taking {drug_name}?",
        "What foods should I avoid while taking {drug_name}?",
        "Does grapefruit juice interact with {drug_name}?",
        "Can I drink alcohol while taking {drug_name}?",
        "Should {drug_name} be taken with meals or on an empty stomach?",
        "Are there dietary restrictions with {drug_name}?",
        "Does {drug_name} interact with caffeine?",
        "Can I eat dairy products with {drug_name}?",
        "Should I limit salt intake while on {drug_name}?",
        "Does {drug_name} affect how I absorb nutrients from food?",
    ],
    "drug_lifestyle_safety": [
        "Is it safe to drive while taking {drug_name}?",
        "Can I exercise regularly while on {drug_name}?",
        "Is it safe to drink alcohol with {drug_name}?",
        "Does {drug_name} affect my ability to work?",
        "Can I take {drug_name} if I'm pregnant or breastfeeding?",
        "Is {drug_name} safe if I want to start a family soon?",
        "Does {drug_name} affect my ability to swim or get wet?",
        "Can I continue my normal activities while on {drug_name}?",
        "Should I limit sun exposure while taking {drug_name}?",
        "Does {drug_name} affect my fertility?",
    ],
    "drug_side_effects": [
        "What are the common side effects of {drug_name}?",
        "Can {drug_name} cause weight gain?",
        "Will {drug_name} make me drowsy?",
        "What serious side effects should I watch for?",
        "Is nausea a common side effect of {drug_name}?",
        "Can {drug_name} cause sexual dysfunction?",
        "Will {drug_name} affect my blood pressure?",
        "What should I do if I experience side effects from {drug_name}?",
        "Are there long-term side effects of {drug_name}?",
        "Will side effects from {drug_name} go away over time?",
    ],
    "patient_indication": [
        "Why was I prescribed {drug_name}?",
        "How will {drug_name} help with my {condition}?",
        "What does it mean that I have {condition}?",
        "Is {drug_name} the right medication for my diagnosis?",
        "What was the reason for starting {drug_name}?",
        "Does {drug_name} cure my {condition} or just manage it?",
        "Will I need to take {drug_name} forever?",
        "Can my {condition} be treated without {drug_name}?",
        "What other treatment options are available for {condition}?",
        "Am I at risk for complications from my {condition}?",
    ],
    "patient_timing": [
        "When should I take {drug_name} during the day?",
        "How many times per day should I take {drug_name}?",
        "What time of day is best for taking {drug_name}?",
        "How long should I wait between doses of {drug_name}?",
        "When should I start taking {drug_name}?",
        "Can I take {drug_name} at night instead of morning?",
        "Do I need to space {drug_name} away from other medications?",
        "What happens if I take {drug_name} at the wrong time?",
        "Should I take {drug_name} with meals as prescribed?",
        "How long will I need to take {drug_name}?",
    ],
    "patient_vitals": [
        "Should I be worried about my recent blood pressure reading?",
        "Is my weight within a healthy range?",
        "What does my HbA1c level mean?",
        "Are my cholesterol numbers good?",
        "Should I be concerned about my heart rate?",
        "What do my recent lab results indicate?",
        "Is my blood sugar level controlled well?",
        "Are my vitals improving since starting {drug_name}?",
        "Should I monitor anything specific at home?",
        "What trends should I watch in my vital signs?",
    ],
    "patient_lifestyle": [
        "What lifestyle changes can help my {condition}?",
        "What diet would be best for managing my {condition}?",
        "How much exercise should I get?",
        "Should I quit smoking for my {condition}?",
        "Can I still do my normal activities?",
        "What lifestyle modifications do you recommend?",
        "How can I improve my health beyond just taking {drug_name}?",
        "What should I avoid in my daily life?",
        "Is stress management important for my {condition}?",
        "What dietary restrictions do I need to follow?",
    ],
    "patient_side_effects": [
        "I'm experiencing fatigue on {drug_name}, is this normal?",
        "Is the headache I'm having from {drug_name}?",
        "My nausea hasn't improved on {drug_name}, what should I do?",
        "Could my symptoms be side effects of {drug_name}?",
        "Should I continue {drug_name} if I'm having side effects?",
        "When will the side effects of {drug_name} go away?",
        "Am I taking {drug_name} correctly if I feel different?",
        "Could my new symptoms be from my {condition}?",
        "Is it normal to feel this way on {drug_name}?",
        "What can I do about the side effects I'm experiencing?",
    ],
    "patient_monitoring": [
        "How often should I have blood work done?",
        "Do I need regular checkups while on {drug_name}?",
        "What should I monitor while taking {drug_name}?",
        "How can I tell if my {condition} is improving?",
        "What warning signs should I watch for?",
        "When should I contact my doctor about my {condition}?",
        "Are there any tests I need regularly?",
        "How is my treatment progress being measured?",
        "Should I keep a symptom diary?",
        "What metrics indicate my treatment is working?",
    ],
}

# Expected mode mapping
CATEGORY_TO_MODE = {
    "drug_general_info": (1, "LLM_only"),
    "drug_mechanism": (1, "LLM_only"),
    "patient_indication": (2, "RAG"),
    "patient_timing": (2, "RAG"),
    "patient_vitals": (2, "RAG"),
    "patient_lifestyle": (2, "RAG"),
    "patient_monitoring": (2, "RAG"),
    "patient_side_effects": (2, "RAG"),
    "drug_interaction": (3, "RAG_KG"),
    "drug_food_interaction": (3, "RAG_KG"),
    "drug_lifestyle_safety": (3, "RAG_KG"),
    "drug_side_effects": (3, "RAG_KG"),
}

# ============================================================================
# Utility functions
# ============================================================================

def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file."""
    records = []
    if file_path.exists():
        with open(file_path, 'r') as f:
            for line in f:
                records.append(json.loads(line.strip()))
    return records

def save_jsonl(file_path: Path, records: List[Dict[str, Any]]) -> None:
    """Save records to JSONL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')

def compute_category_distribution(questions: List[Dict]) -> Dict[str, float]:
    """Compute category distribution as proportions."""
    category_counts = defaultdict(int)
    for q in questions:
        category_counts[q['category']] += 1

    total = sum(category_counts.values())
    return {cat: count / total for cat, count in category_counts.items()}

def generate_patient(patient_id: str, existing_patients: List[Dict]) -> Dict[str, Any]:
    """Generate a single synthetic patient."""
    gender = random.choice(["Male", "Female"])
    name = f"{random.choice(FIRST_NAMES[gender])} {random.choice(LAST_NAMES)}"
    age = random.randint(25, 85)

    # Date of birth
    dob_date = datetime.now() - timedelta(days=age * 365.25 + random.randint(0, 365))
    dob = dob_date.strftime("%Y-%m-%d")

    # Select 1-4 active conditions
    num_conditions = random.choices([1, 2, 3, 4], weights=[0.4, 0.35, 0.2, 0.05])[0]
    conditions = random.sample(list(CONDITION_MEDICATIONS.keys()), min(num_conditions, len(CONDITION_MEDICATIONS)))

    active_conditions = [
        {
            "condition": cond,
            "diagnosed": str(random.randint(2010, 2024))
        }
        for cond in conditions
    ]

    # Medications based on conditions
    medications = []
    med_set = set()
    for cond in conditions:
        # 1-2 medications per condition
        num_meds = random.randint(1, 2)
        cond_meds = random.sample(CONDITION_MEDICATIONS[cond], min(num_meds, len(CONDITION_MEDICATIONS[cond])))
        for drug_name, dosage, frequency in cond_meds:
            if drug_name not in med_set:
                med_set.add(drug_name)
                medications.append({
                    "drug": drug_name,
                    "dosage": dosage,
                    "frequency": frequency,
                    "indication": f"{cond} management",
                    "start_date": f"{random.randint(2015, 2024)}-{random.randint(1, 12):02d}",
                    "duration": "Ongoing",
                    "notes": f"Effective for {cond}"
                })

    # Allergies (16-30% of patients have at least one)
    allergies = []
    if random.random() < 0.25:
        num_allergies = random.randint(1, 2)
        allergies = random.sample(ALLERGIES_POOL, min(num_allergies, len(ALLERGIES_POOL)))

    # Recent vitals (coherent with conditions)
    bp_base = (120, 80)
    heart_rate = random.randint(60, 85)

    # Adjust BP for hypertension
    if any("Hypertension" in c["condition"] for c in active_conditions):
        bp_base = (random.randint(130, 145), random.randint(80, 92))
    else:
        bp_base = (random.randint(110, 130), random.randint(70, 85))

    weight_kg = random.randint(55, 110)
    height_cm = random.randint(160, 185)
    bmi = round(weight_kg / (height_cm / 100) ** 2, 1)

    vitals = {
        "date": (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d"),
        "blood_pressure": f"{bp_base[0]}/{bp_base[1]}",
        "heart_rate": heart_rate,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "bmi": bmi,
    }

    # Add condition-specific vitals
    if any("Diabetes" in c["condition"] for c in active_conditions):
        vitals["hba1c"] = round(random.uniform(6.5, 9.0), 1)
        vitals["fasting_glucose"] = random.randint(100, 200)
    if any("Hyperlipidemia" in c["condition"] or "Coronary" in c["condition"] for c in active_conditions):
        vitals["ldl"] = random.randint(60, 150)
        vitals["hdl"] = random.randint(35, 60)
        vitals["triglycerides"] = random.randint(100, 300)
    if any("eGFR" in c["condition"] or "Diabetes" in c["condition"] for c in active_conditions):
        vitals["egfr"] = random.randint(60, 120)

    # Lifestyle
    lifestyle = {
        "smoking": random.choice(["Never", "Former (quit 2015+)", "Current"]),
        "alcohol": random.choice(["Never", "Rare (special occasions)", "Occasional", "Moderate", "Heavy"]),
        "exercise": random.choice(["Sedentary", "Light (walks occasionally)", "Moderate (3-4x/week)", "Vigorous (5+x/week)"]),
        "diet": random.choice(["Standard American", "Mediterranean", "DASH", "Vegetarian", "Low-carb"]),
    }

    # Narrative
    cond_str = "; ".join([c["condition"] for c in active_conditions])
    med_str = "; ".join([m["drug"] for m in medications]) if medications else "no medications"
    narrative = f"{name} is a {age}-year-old {gender.lower()} with {cond_str}. "
    narrative += f"Current medications: {med_str}. "
    if allergies:
        allergy_str = "; ".join([f"{a['substance']} ({a['reaction']})" for a in allergies])
        narrative += f"Allergies: {allergy_str}. "
    narrative += f"Lives a {lifestyle['exercise'].lower()} lifestyle. "
    narrative += f"Recent vitals show BP {vitals['blood_pressure']}, HR {vitals['heart_rate']}, BMI {vitals['bmi']}."

    return {
        "patient_id": patient_id,
        "synthetic": True,
        "name": name,
        "age": age,
        "gender": gender,
        "dob": dob,
        "active_conditions": active_conditions,
        "past_medical_history": [],
        "allergies": allergies,
        "medications": medications,
        "recent_vitals": vitals,
        "lifestyle": lifestyle,
        "narrative": narrative,
    }

def generate_questions_for_distribution(
    all_patients: List[Dict],
    target_distribution: Dict[str, float],
    total_questions: int,
    question_id_start: int,
) -> Tuple[List[Dict], int]:
    """
    Generate questions matching target category distribution.
    Return (questions, next_question_id).
    """
    questions = []

    # Determine target counts per category
    category_counts = {cat: int(round(target_distribution.get(cat, 0) * total_questions))
                       for cat in CATEGORY_TO_MODE.keys()}

    # Adjust for rounding errors
    actual_total = sum(category_counts.values())
    if actual_total != total_questions:
        largest_cat = max(category_counts, key=category_counts.get)
        category_counts[largest_cat] += (total_questions - actual_total)

    qid = question_id_start

    for category in sorted(CATEGORY_TO_MODE.keys()):
        target_count = category_counts[category]
        templates = QUESTION_TEMPLATES[category]
        expected_mode, expected_mode_name = CATEGORY_TO_MODE[category]

        for _ in range(target_count):
            # Select patient: ~80% from new (P051+), ~20% from old (P001-P050)
            if random.random() < 0.8:
                # New patients
                patient = random.choice(all_patients[50:])
            else:
                # Old patients
                patient = random.choice(all_patients[:50])

            # Select template and fill
            template = random.choice(templates)

            # Get a drug from patient's medications
            if patient["medications"]:
                drug_name = random.choice(patient["medications"])["drug"]
            else:
                drug_name = random.choice(["Aspirin", "Ibuprofen", "Acetaminophen"])

            # Get a condition
            if patient["active_conditions"]:
                condition = random.choice(patient["active_conditions"])["condition"]
            else:
                condition = "medical condition"

            question_text = template.format(drug_name=drug_name, condition=condition)

            # Generate rationale
            mode_desc = {1: "generic pharmacology", 2: "patient-specific context", 3: "drug interaction knowledge"}
            rationale = f"Requires {mode_desc.get(expected_mode, 'knowledge')} for answering"

            questions.append({
                "question_id": f"Q{qid:04d}",
                "patient_id": patient["patient_id"],
                "question": question_text,
                "category": category,
                "expected_mode": expected_mode,
                "expected_mode_name": expected_mode_name,
                "rationale": rationale,
            })
            qid += 1

    return questions, qid

# ============================================================================
# Main execution
# ============================================================================

def main():
    print("=" * 80)
    print("SYNTHETIC PATIENT DATASET GENERATOR v2")
    print("=" * 80)

    # Load existing data
    print("\n[1] Loading existing data...")
    existing_patients = load_jsonl(PATIENTS_FILE)
    existing_questions = load_jsonl(QUESTIONS_FILE)
    existing_test_questions = load_jsonl(TEST_QUESTIONS_FILE)

    print(f"   Existing: {len(existing_patients)} patients, "
          f"{len(existing_questions)} train questions, "
          f"{len(existing_test_questions)} test questions")

    # Compute category distribution from combined questions
    print("\n[2] Computing category distribution...")
    all_existing_questions = existing_questions + existing_test_questions
    target_distribution = compute_category_distribution(all_existing_questions)

    print("   Target category distribution:")
    for cat in sorted(CATEGORY_TO_MODE.keys()):
        pct = target_distribution.get(cat, 0) * 100
        print(f"     {cat:30s}: {pct:5.1f}%")

    # Generate new patients
    print("\n[3] Generating 350 new patients...")
    new_patients = []
    last_patient_id = int(existing_patients[-1]["patient_id"][1:])

    for i in range(350):
        patient_id = f"P{last_patient_id + i + 1:03d}"
        patient = generate_patient(patient_id, existing_patients)
        new_patients.append(patient)
        if (i + 1) % 50 == 0:
            print(f"     Generated {i + 1} patients")

    all_patients = existing_patients + new_patients
    print(f"   Total patients: {len(all_patients)}")

    # Generate new questions (train + test)
    # Target: 1600 total train = 250 existing + 1350 new
    # Target: 400 total test = 100 existing + 300 new
    new_train_count = 1600 - len(existing_questions)
    new_test_count = 400 - len(existing_test_questions)

    print(f"\n[4] Generating {new_train_count} new training questions (target 1600 total)...")
    last_question_id = int(existing_questions[-1]["question_id"][1:])
    train_questions, next_qid = generate_questions_for_distribution(
        all_patients, target_distribution, new_train_count, last_question_id + 1
    )
    print(f"   Generated {len(train_questions)} new training questions")

    print(f"\n[5] Generating {new_test_count} new test questions (target 400 total)...")
    test_questions, next_qid = generate_questions_for_distribution(
        all_patients, target_distribution, new_test_count, next_qid
    )
    print(f"   Generated {len(test_questions)} new test questions")

    # Combine all questions
    all_new_questions = existing_questions + train_questions
    all_new_test_questions = existing_test_questions + test_questions

    # Save files
    print("\n[6] Saving output files...")
    save_jsonl(PATIENTS_V2_FILE, all_patients)
    print(f"   Saved {len(all_patients)} patients to {PATIENTS_V2_FILE}")

    save_jsonl(QUESTIONS_V2_FILE, all_new_questions)
    print(f"   Saved {len(all_new_questions)} training questions to {QUESTIONS_V2_FILE}")

    save_jsonl(TEST_QUESTIONS_V2_FILE, all_new_test_questions)
    print(f"   Saved {len(all_new_test_questions)} test questions to {TEST_QUESTIONS_V2_FILE}")

    # Verify and report
    print("\n[7] Verification and reporting...")

    # Count lines
    patients_v2 = load_jsonl(PATIENTS_V2_FILE)
    questions_v2 = load_jsonl(QUESTIONS_V2_FILE)
    test_questions_v2 = load_jsonl(TEST_QUESTIONS_V2_FILE)

    print(f"\n   FILE COUNTS:")
    print(f"     patients_v2.jsonl:       {len(patients_v2)} records (expected 400) {'✓' if len(patients_v2) == 400 else '✗'}")
    print(f"     questions_v2.jsonl:      {len(questions_v2)} records (expected 1600) {'✓' if len(questions_v2) == 1600 else '✗'}")
    print(f"     test_questions_v2.jsonl: {len(test_questions_v2)} records (expected 400) {'✓' if len(test_questions_v2) == 400 else '✗'}")

    # Verify no ID collisions
    patient_ids = set(p["patient_id"] for p in patients_v2)
    assert len(patient_ids) == len(patients_v2), "Patient ID collision detected!"
    print(f"\n   All {len(patient_ids)} patient IDs are unique ✓")

    question_ids = set(q["question_id"] for q in questions_v2)
    test_ids = set(q["question_id"] for q in test_questions_v2)
    assert len(question_ids & test_ids) == 0, "Question ID collision between train and test!"
    print(f"   No question ID collisions between train ({len(question_ids)}) and test ({len(test_ids)}) ✓")

    # Category distribution verification
    print(f"\n   CATEGORY DISTRIBUTION:")
    print(f"   {'Category':<30s} {'Original %':>10s} {'Train v2 %':>10s} {'Test v2 %':>10s} {'Combined %':>10s}")
    print(f"   {'-' * 70}")

    all_v2_questions = questions_v2 + test_questions_v2
    new_dist = compute_category_distribution(all_v2_questions)
    v2_train_dist = compute_category_distribution(questions_v2)
    v2_test_dist = compute_category_distribution(test_questions_v2)

    for cat in sorted(CATEGORY_TO_MODE.keys()):
        orig_pct = target_distribution.get(cat, 0) * 100
        train_pct = v2_train_dist.get(cat, 0) * 100
        test_pct = v2_test_dist.get(cat, 0) * 100
        combined_pct = new_dist.get(cat, 0) * 100
        diff = abs(combined_pct - orig_pct)
        marker = "✓" if diff <= 3 else "!"

        print(f"   {cat:<30s} {orig_pct:>9.1f}% {train_pct:>9.1f}% {test_pct:>9.1f}% {combined_pct:>9.1f}% {marker}")

    # Sample records
    print(f"\n   SAMPLE PATIENT (from v2):")
    sample_patient = patients_v2[50]  # First new patient
    print(f"     ID: {sample_patient['patient_id']}")
    print(f"     Name: {sample_patient['name']}")
    print(f"     Age: {sample_patient['age']}, Gender: {sample_patient['gender']}")
    print(f"     Conditions: {[c['condition'] for c in sample_patient['active_conditions']]}")
    print(f"     Medications: {[m['drug'] for m in sample_patient['medications']]}")
    print(f"     Allergies: {[a['substance'] for a in sample_patient['allergies']]}")

    print(f"\n   SAMPLE TRAINING QUESTION (from v2):")
    sample_q = questions_v2[250]
    print(f"     ID: {sample_q['question_id']}")
    print(f"     Patient: {sample_q['patient_id']}")
    print(f"     Category: {sample_q['category']}")
    print(f"     Mode: {sample_q['expected_mode_name']}")
    print(f"     Question: {sample_q['question']}")

    print(f"\n   SAMPLE TEST QUESTION (from v2):")
    sample_t = test_questions_v2[50]
    print(f"     ID: {sample_t['question_id']}")
    print(f"     Patient: {sample_t['patient_id']}")
    print(f"     Category: {sample_t['category']}")
    print(f"     Mode: {sample_t['expected_mode_name']}")
    print(f"     Question: {sample_t['question']}")

    print("\n" + "=" * 80)
    print("GENERATION COMPLETE ✓")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - {PATIENTS_V2_FILE.relative_to(PROJECT_ROOT)}")
    print(f"  - {QUESTIONS_V2_FILE.relative_to(PROJECT_ROOT)}")
    print(f"  - {TEST_QUESTIONS_V2_FILE.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    main()
