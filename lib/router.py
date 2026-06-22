"""
Loads the trained MLP router and produces predictions for live questions.

Feature extraction matches `Patient_Portal_Router.ipynb` exactly so the
saved weights and scaler stay valid.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Categories — must match training-time order exactly
# ---------------------------------------------------------------------------

CATEGORY_ORDER = [
    "drug_general_info", "drug_mechanism", "drug_interaction", "drug_food_interaction",
    "drug_lifestyle_safety", "drug_side_effects",
    "patient_indication", "patient_timing", "patient_vitals", "patient_lifestyle",
    "patient_side_effects", "patient_monitoring",
]
CATEGORY_TO_IDX = {c: i for i, c in enumerate(CATEGORY_ORDER)}

SAFETY_RELATIONS = {
    "drug-drug interaction", "synergistic interaction",
    "side effect", "contraindication",
}


# ---------------------------------------------------------------------------
# Lightweight rule-based category detector for live questions
# ---------------------------------------------------------------------------

CATEGORY_RULES = [
    ("drug_food_interaction", [r"\bgrapefruit\b", r"\bcoffee\b", r"\bcaffeine\b",
                                r"\bfood\b.*\binteract", r"\bcalcium\b.*\bwith\b"]),
    ("drug_lifestyle_safety", [r"\balcohol\b", r"\bdrink\b", r"\bwine\b", r"\bbeer\b",
                                r"\bsafe to drink\b", r"\bcan i drink\b",
                                r"\bsmok", r"\bpregnan", r"\bbreastfeed"]),
    ("drug_interaction",      [r"\binteract", r"\bwith my\b", r"\btake .* with\b",
                                r"\bibuprofen\b", r"\baspirin\b.*\bwith\b",
                                r"\btogether\b", r"\bcombine\b"]),
    ("drug_mechanism",        [r"\bhow does\b", r"\bmechanism\b", r"\bwork\b"]),
    ("drug_side_effects",     [r"\bside effects?\b", r"\badverse\b",
                                r"\bdrowsy\b", r"\bnausea\b"]),
    ("drug_general_info",     [r"\bwhat is\b", r"\bwhat does\b", r"\bwhat'?s\b",
                                r"\bcommon\b.*\b(drug|med)"]),
    ("patient_timing",        [r"\bwhen should\b", r"\bwhat time\b", r"\bhow often\b",
                                r"\bevery\b.*\bhours\b"]),
    ("patient_indication",    [r"\bwhy was i\b", r"\bwhy am i\b", r"\bwhy do i\b",
                                r"\bwhy did\b", r"\bwhy is\b"]),
    ("patient_vitals",        [r"\bblood pressure\b", r"\bhba1c\b", r"\bcholesterol\b",
                                r"\bmy.*levels?\b", r"\bmy.*lab"]),
    ("patient_monitoring",    [r"\bmonitor\b", r"\btrack\b", r"\btell if\b.*\bworking\b",
                                r"\bimprove"]),
    ("patient_side_effects",  [r"\bnoticing\b", r"\bsince starting\b",
                                r"\bcould it be\b.*\bmedication\b"]),
    ("patient_lifestyle",     [r"\bexercise\b", r"\bdiet\b", r"\bsleep\b",
                                r"\btravel\b", r"\bvacation\b"]),
]


def predict_category(question: str) -> str:
    q = question.lower()
    for cat, patterns in CATEGORY_RULES:
        for p in patterns:
            if re.search(p, q):
                return cat
    return "drug_general_info"  # default fallback


# ---------------------------------------------------------------------------
# Feature extraction — must match training exactly
# ---------------------------------------------------------------------------

def get_retrieval_features(mode_2_record: dict | None) -> tuple[float, float, float, float]:
    if not mode_2_record:
        return 0.0, 0.0, 0.0, 0.0
    chunks = mode_2_record.get("retrieved_chunks", [])
    scores = [c.get("score", 0.0) for c in chunks]
    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) >= 2 else 0.0
    return top1, top2, top1 - top2, float(np.mean(scores[:3])) if scores else 0.0


def get_kg_features(mode_3_record: dict | None) -> tuple[int, int]:
    if not mode_3_record:
        return 0, 0
    n = mode_3_record.get("n_kg_triples", 0)
    triples = mode_3_record.get("kg_triples", [])
    has_safety = 0
    for t in triples:
        if any(rel in t.lower() for rel in SAFETY_RELATIONS):
            has_safety = 1
            break
    return n, has_safety


def get_patient_features(patient: dict) -> tuple[int, int, int]:
    return (
        len(patient.get("medications", [])),
        len(patient.get("active_conditions", [])),
        1 if patient.get("allergies") else 0,
    )


def count_drugs_in_question(question: str, patient: dict) -> int:
    q = question.lower()
    n = 0
    for m in patient.get("medications", []):
        first = m["drug"].split()[0].lower()
        if first in q:
            n += 1
    return n


def build_features(
    question: str,
    category: str,
    patient: dict,
    mode_2_record: dict | None,
    mode_3_record: dict | None,
) -> np.ndarray:
    type_oh = [0] * len(CATEGORY_ORDER)
    if category in CATEGORY_TO_IDX:
        type_oh[CATEGORY_TO_IDX[category]] = 1

    top1, top2, gap, mean3 = get_retrieval_features(mode_2_record)
    n_kg, has_safety = get_kg_features(mode_3_record)
    n_meds, n_cond, has_allerg = get_patient_features(patient)
    q_len = len(question.split())
    n_drugs_q = count_drugs_in_question(question, patient)

    feat = type_oh + [
        top1, top2, gap, mean3,
        n_kg, has_safety,
        n_meds, n_cond, has_allerg,
        q_len, n_drugs_q,
    ]
    return np.array(feat, dtype=np.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RouterMLP(nn.Module):
    def __init__(
        self, input_dim: int = 23, hidden1: int = 32,
        hidden2: int = 16, n_classes: int = 3, dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),   nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Router:
    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self.weights_path = project_dir / "patient_router" / "router_mlp.pt"
        self.scaler_path = project_dir / "patient_router" / "scaler.pkl"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model: RouterMLP | None = None
        self.scaler: Any | None = None
        self.error: str | None = None

    def load(self) -> None:
        if not self.weights_path.exists() or not self.scaler_path.exists():
            self.error = (
                f"Router files missing. Expected:\n  {self.weights_path}\n  {self.scaler_path}"
            )
            return
        try:
            self.model = RouterMLP(input_dim=23).to(self.device)
            state = torch.load(self.weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            with self.scaler_path.open("rb") as f:
                self.scaler = pickle.load(f)
        except Exception as e:
            self.error = f"Router load failed: {e}"
            self.model, self.scaler = None, None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.scaler is not None

    def predict(
        self, question: str, category: str, patient: dict,
        mode_2_record: dict | None, mode_3_record: dict | None,
    ) -> dict:
        feat = build_features(question, category, patient, mode_2_record, mode_3_record)
        if not self.ready:
            return {
                "predicted_mode": 2,
                "confidence": 0.0,
                "probs": [0.0, 1.0, 0.0],
                "category": category,
                "fallback": True,
            }
        feat_scaled = self.scaler.transform(feat.reshape(1, -1)).astype(np.float32)
        with torch.no_grad():
            logits = self.model(torch.tensor(feat_scaled).to(self.device))
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            pred = int(np.argmax(probs)) + 1
        return {
            "predicted_mode": pred,
            "confidence": float(probs.max()),
            "probs": [float(p) for p in probs],
            "category": category,
            "fallback": False,
        }
