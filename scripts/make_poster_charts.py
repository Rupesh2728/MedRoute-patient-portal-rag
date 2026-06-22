"""Generate poster charts.

Data-analysis charts (describe the dataset):
  A — question-category distribution (train + test)
  B — top prescribed drugs across cohort
  C — most common active conditions across cohort

Results charts (describe model performance):
  D — per-mode accuracy by category
  E — router confusion matrix
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "poster_charts"
OUT_DIR.mkdir(exist_ok=True)

C_DARK = "#005BBB"
C_MID = "#4A90D9"
C_LIGHT = "#9DC3E6"
C_ACCENT = "#E76F51"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


# ---- Load data ----------------------------------------------------------
patients = load_jsonl(PROJECT / "synthetic_patients" / "patients.jsonl")
train_q = load_jsonl(PROJECT / "synthetic_patients" / "questions.jsonl")
test_q = load_jsonl(PROJECT / "synthetic_patients" / "test_questions.jsonl")
gt = load_jsonl(PROJECT / "synthetic_patients" / "patient_portal_ground_truth_v2.jsonl")
preds = load_jsonl(PROJECT / "patient_router" / "router_predictions.jsonl")

CATEGORY_DISPLAY = {
    "drug_general_info":     "Drug: general info",
    "drug_mechanism":        "Drug: mechanism",
    "drug_interaction":      "Drug: interaction",
    "drug_food_interaction": "Drug: food",
    "drug_lifestyle_safety": "Drug: lifestyle",
    "drug_side_effects":     "Drug: side effects",
    "patient_indication":    "Patient: indication",
    "patient_timing":        "Patient: timing",
    "patient_vitals":        "Patient: vitals",
    "patient_lifestyle":     "Patient: lifestyle",
    "patient_side_effects":  "Patient: side effects",
    "patient_monitoring":    "Patient: monitoring",
}
ORDER = list(CATEGORY_DISPLAY.keys())


# =========================================================================
# CHART A — Question-category distribution
# =========================================================================
train_counts = Counter(q["category"] for q in train_q)
test_counts = Counter(q["category"] for q in test_q)
train_vals = [train_counts.get(c, 0) for c in ORDER]
test_vals = [test_counts.get(c, 0) for c in ORDER]
labels = [CATEGORY_DISPLAY[c] for c in ORDER]

fig, ax = plt.subplots(figsize=(8.5, 5.0))
y = np.arange(len(ORDER))
ax.barh(y, train_vals, color=C_DARK, label="Train")
ax.barh(y, test_vals, left=train_vals, color=C_LIGHT, label="Test")
ax.set_yticks(y); ax.set_yticklabels(labels); ax.invert_yaxis()
ax.set_xlabel("Number of questions")
ax.set_title("Chart A: Question categories")
ax.legend(loc="lower right", frameon=False)
ax.set_xlim(0, max(t + e for t, e in zip(train_vals, test_vals)) + 6)
fig.tight_layout()
fig.savefig(OUT_DIR / "chart_a_category_distribution.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)


# =========================================================================
# CHART B — Top prescribed drugs across the cohort
# =========================================================================
drug_counts: Counter = Counter()
for p in patients:
    for m in p.get("medications", []):
        # use first word as the canonical name (lowercased)
        name = m["drug"].split()[0].strip().lower()
        drug_counts[name] += 1
top_drugs = drug_counts.most_common(10)
drug_names = [d.capitalize() for d, _ in top_drugs]
drug_n = [n for _, n in top_drugs]

fig, ax = plt.subplots(figsize=(7.5, 4.5))
y = np.arange(len(drug_names))
ax.barh(y, drug_n, color=C_DARK)
ax.set_yticks(y); ax.set_yticklabels(drug_names); ax.invert_yaxis()
ax.set_xlabel("Number of patients prescribed")
ax.set_title("Chart B: Top 10 prescribed drugs")
fig.tight_layout()
fig.savefig(OUT_DIR / "chart_b_top_drugs.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)


# =========================================================================
# CHART C — Most common active conditions across the cohort
# =========================================================================
cond_counts: Counter = Counter()
for p in patients:
    for c in p.get("active_conditions", []):
        name = c["condition"].strip().lower()
        cond_counts[name] += 1
top_conds = cond_counts.most_common(10)
cond_names = [c.title() for c, _ in top_conds]
cond_n = [n for _, n in top_conds]

fig, ax = plt.subplots(figsize=(7.5, 4.5))
y = np.arange(len(cond_names))
ax.barh(y, cond_n, color=C_MID)
ax.set_yticks(y); ax.set_yticklabels(cond_names); ax.invert_yaxis()
ax.set_xlabel("Number of patients with condition")
ax.set_title("Chart C: Top 10 active conditions")
fig.tight_layout()
fig.savefig(OUT_DIR / "chart_c_top_conditions.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)


# =========================================================================
# RESULTS CHART D — Per-mode accuracy by category
# =========================================================================
mode_correct = {1: defaultdict(int), 2: defaultdict(int), 3: defaultdict(int)}
totals = defaultdict(int)
for r in gt:
    cat = r["category"]
    totals[cat] += 1
    for m in (1, 2, 3):
        if r.get(f"mode_{m}_correct") == 1:
            mode_correct[m][cat] += 1
acc = np.zeros((3, len(ORDER)))
for j, cat in enumerate(ORDER):
    n = totals.get(cat, 0)
    for i, m in enumerate([1, 2, 3]):
        acc[i, j] = (mode_correct[m][cat] / n * 100) if n else 0

fig, ax = plt.subplots(figsize=(9.0, 4.5))
im = ax.imshow(acc, aspect="auto", cmap="Blues", vmin=0, vmax=100)
ax.set_xticks(np.arange(len(ORDER)))
ax.set_xticklabels(labels, rotation=40, ha="right")
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["Mode 1\nLLM only", "Mode 2\n+ RAG", "Mode 3\n+ RAG + KG"])
for i in range(3):
    for j in range(len(ORDER)):
        v = acc[i, j]
        color = "white" if v >= 60 else "#222"
        ax.text(j, i, f"{int(round(v))}", ha="center", va="center",
                color=color, fontsize=9)
ax.set_title("Chart D: Per-mode accuracy by category")
cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("Accuracy (%)")
fig.tight_layout()
fig.savefig(OUT_DIR / "chart_d_per_category_accuracy.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)


# =========================================================================
# RESULTS CHART E — Router confusion matrix
# =========================================================================
cm = np.zeros((3, 3), dtype=int)
for p in preds:
    cm[p["expected_mode"] - 1, p["predicted_mode"] - 1] += 1
row_totals = cm.sum(axis=1, keepdims=True)
cm_pct = cm / np.where(row_totals == 0, 1, row_totals) * 100

fig, ax = plt.subplots(figsize=(5.5, 4.6))
ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
ax.set_xticklabels(["Mode 1", "Mode 2", "Mode 3"])
ax.set_yticklabels(["Mode 1", "Mode 2", "Mode 3"])
ax.set_xlabel("Router prediction")
ax.set_ylabel("Expected (winning) mode")
for i in range(3):
    for j in range(3):
        v = cm_pct[i, j]
        color = "white" if v >= 50 else "#222"
        ax.text(j, i, f"{int(round(v))}%", ha="center", va="center",
                color=color, fontsize=12, fontweight="bold")
ax.set_title("Chart E: Router confusion matrix")
fig.tight_layout()
fig.savefig(OUT_DIR / "chart_e_router_confusion.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)


# =========================================================================
# Print cohort stats so we can use them in poster prose
# =========================================================================
n_patients = len(patients)
n_meds = [len(p.get("medications", [])) for p in patients]
n_conds = [len(p.get("active_conditions", [])) for p in patients]
n_with_allergy = sum(1 for p in patients if p.get("allergies"))

print("Saved:")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f" - {f.name}")
print()
print("=== Cohort stats ===")
print(f"  patients:           {n_patients}")
print(f"  median meds:        {int(np.median(n_meds))}  (mean {np.mean(n_meds):.1f})")
print(f"  median conditions:  {int(np.median(n_conds))}  (mean {np.mean(n_conds):.1f})")
print(f"  with allergy:       {n_with_allergy} ({100 * n_with_allergy / n_patients:.0f}%)")
print()
print("=== Top 10 conditions ===")
for c, n in top_conds:
    print(f"  {n:4d}  {c}")
print()
print("=== Top 10 drugs ===")
for d, n in top_drugs:
    print(f"  {n:4d}  {d}")
