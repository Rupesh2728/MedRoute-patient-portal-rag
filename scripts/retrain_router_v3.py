"""
Retrain the MLP router with MedlinePlus retrieval features (v3 = 27 features).

In this we are using existing patient retrieval / KG features from the saved
patient_portal_responses_v2.jsonl and ground_truth labels. Only adds the
4 new MedlinePlus retrieval features per question. 

Outputs:
  patient_router/router_mlp_v3.pt
  patient_router/scaler_v3.pkl
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parent.parent

# Make sure we can import the streamlit_app lib
sys.path.insert(0, str(PROJECT / "streamlit_app"))
from lib.pipeline import LiveBackend  # noqa: E402
from lib.router import (  # noqa: E402
    CATEGORY_ORDER, RouterMLP, build_features, predict_category,
)

DATA_DIR = PROJECT / "synthetic_patients"
ROUTER_DIR = PROJECT / "patient_router"


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    print("Loading data…")
    patients = {p["patient_id"]: p
                for p in load_jsonl(DATA_DIR / "patients_v2.jsonl")}


    train_q = load_jsonl(DATA_DIR / "questions_v2.jsonl")
    test_q = load_jsonl(DATA_DIR / "test_questions_v2.jsonl")
    questions = train_q + test_q


    responses_path_v3 = DATA_DIR / "patient_portal_responses_v3_2000.jsonl"
    responses_path_v2 = DATA_DIR / "patient_portal_responses_v2.jsonl"
    responses_path = (responses_path_v3 if responses_path_v3.exists()
                       else responses_path_v2)
    responses = {r["question_id"]: r for r in
                 (load_jsonl(responses_path) if responses_path.exists() else [])}
    print(f"  using responses: {responses_path.name}")

    # Prefer STRICT ground truth (medically-aware rubric) when present.
    gt_path_strict = DATA_DIR / "patient_portal_ground_truth_v3_2000_STRICT.jsonl"
    gt_path_v3 = DATA_DIR / "patient_portal_ground_truth_v3_2000.jsonl"
    gt_path_v2 = DATA_DIR / "patient_portal_ground_truth_v2.jsonl"
    gt_path = (gt_path_strict if gt_path_strict.exists()
               else (gt_path_v3 if gt_path_v3.exists() else gt_path_v2))
    gt = {g["question_id"]: g for g in
          (load_jsonl(gt_path) if gt_path.exists() else [])}
    print(f"  using ground truth: {gt_path.name}")

    print(f"  patients: {len(patients)}  questions: {len(questions)} "
          f"({len(train_q)} train + {len(test_q)} test)  "
          f"saved responses: {len(responses)}  real-labels: {len(gt)}")

    print("\nLoading live backend (MedCPT + MedlinePlus + KG)…")
    backend = LiveBackend(project_dir=PROJECT)
    backend.load()
    if not backend.encoders_ready:
        print(f"ERROR: encoders not loaded: {backend.warnings}")
        sys.exit(1)
    if not backend.medline_ready:
        print(f"ERROR: MedlinePlus not loaded: {backend.warnings}")
        sys.exit(1)
    print(f"  status: {backend.status_summary()}")

    print("\nBuilding 27-feature vectors…")
    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    qids_used: list[str] = []
    skipped = 0

    t0 = time.time()
    used_real_label = 0
    used_auto_label = 0
    for q in tqdm(questions, desc="features"):
        qid = q["question_id"]
        patient = patients.get(q["patient_id"])
        if patient is None:
            skipped += 1
            continue

        # ---- Patient retrieval + KG features --------------------------------
        resp = responses.get(qid)
        if resp:
            m2_existing = resp.get("mode_2", {})
            m3 = resp.get("mode_3", {})
        else:
            # No saved response for this v2 question — compute fresh
            reranked, all_triples, m2_partial, m3_partial = (
                backend.prepare_features(q["question"], patient)
            )
            m2_existing = m2_partial
            m3 = m3_partial

        # ---- Fresh MedlinePlus retrieval (the new signal) -------------------
        cand_m = backend.retrieve_medlineplus_chunks(q["question"], top_k=10)
        reranked_m = (
            backend.rerank_chunks(q["question"], cand_m, top_k=5) if cand_m else []
        )
        med_chunks = [
            {
                "title": c["title"], "url": c.get("url", ""),
                "score": c.get("cross_encoder_score", c["score"]),
                "text": c["text"][:200],
            }
            for c in reranked_m
        ]
        m2_combined = dict(m2_existing)
        m2_combined["medlineplus_chunks"] = med_chunks
        m2_combined["n_medlineplus_chunks"] = len(med_chunks)

        # ---- Category --------------------------------------------------------
        category = q.get("category") or predict_category(q["question"])
        feat = build_features(
            question=q["question"], category=category, patient=patient,
            mode_2_record=m2_combined, mode_3_record=m3,
        )

        winner = None
        if qid in gt and gt[qid].get("winning_mode") in (1, 2, 3):
            winner = gt[qid]["winning_mode"]
            used_real_label += 1
        elif q.get("expected_mode") in (1, 2, 3):
            winner = q["expected_mode"]
            used_auto_label += 1

        if winner not in (1, 2, 3):
            skipped += 1
            continue

        X_rows.append(feat)
        y_rows.append(int(winner) - 1)
        qids_used.append(qid)

    print(f"\n  labels: {used_real_label} real (winning_mode)  "
          f"{used_auto_label} auto (expected_mode)")

    print(f"\n  built features for {len(X_rows)} questions  (skipped {skipped})  "
          f"({time.time() - t0:.1f}s)")

    X = np.stack(X_rows).astype(np.float32)
    y = np.array(y_rows, dtype=np.int64)
    print(f"  X shape: {X.shape}  ({X.shape[1]} features)")
    print(f"  label distribution: {Counter(int(v) + 1 for v in y)}")

    print("\n5-fold stratified CV…")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs: list[float] = []

    DEVICE = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"  device: {DEVICE}")

    fold_macro_f1s: list[float] = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        sc = StandardScaler().fit(X[tr_idx])
        X_tr = sc.transform(X[tr_idx]).astype(np.float32)
        X_va = sc.transform(X[va_idx]).astype(np.float32)

        # Class weights — inverse frequency, balances majority Mode 1 vs minority Mode 2/3
        cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y[tr_idx])
        cw_t = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
        if fold == 0:
            print(f"  class weights: M1={cw[0]:.2f}  M2={cw[1]:.2f}  M3={cw[2]:.2f}")

        model = RouterMLP(input_dim=X.shape[1]).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss(weight=cw_t)
        Xt = torch.tensor(X_tr).to(DEVICE)
        yt = torch.tensor(y[tr_idx]).to(DEVICE)
        Xv = torch.tensor(X_va).to(DEVICE)
        yv = torch.tensor(y[va_idx]).to(DEVICE)

        best_va = 0.0
        best_state = None
        bad = 0
        for ep in range(300):
            model.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), 32):
                idx = perm[i: i + 32]
                opt.zero_grad()
                loss = crit(model(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                preds = model(Xv).argmax(-1)
                va = (preds == yv).float().mean().item()
            if va > best_va:
                best_va = va
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= 40:
                    break
        model.load_state_dict(best_state)
        fold_accs.append(best_va)
        # Macro-F1 on the validation fold
        with torch.no_grad():
            preds_va = model(Xv).argmax(-1).cpu().numpy()
        macro_f1 = f1_score(y[va_idx], preds_va, average="macro", labels=[0, 1, 2])
        fold_macro_f1s.append(macro_f1)
        print(f"  fold {fold}: val_acc = {best_va * 100:.1f}%  macro_F1 = {macro_f1 * 100:.1f}%")

    cv_mean = float(np.mean(fold_accs))
    cv_std = float(np.std(fold_accs))
    f1_mean = float(np.mean(fold_macro_f1s))
    f1_std = float(np.std(fold_macro_f1s))
    print(f"\nCV accuracy: {cv_mean * 100:.2f}% +/- {cv_std * 100:.2f}%")
    print(f"CV macro-F1: {f1_mean * 100:.2f}% +/- {f1_std * 100:.2f}%")
    print("(macro-F1 < 50% means the router is collapsing to majority class)")


    print("\nTraining final model on all data…")
    final_scaler = StandardScaler().fit(X)
    Xs = final_scaler.transform(X).astype(np.float32)
    final_cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y)
    final_cw_t = torch.tensor(final_cw, dtype=torch.float32).to(DEVICE)
    model = RouterMLP(input_dim=X.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=final_cw_t)
    Xt = torch.tensor(Xs).to(DEVICE)
    yt = torch.tensor(y).to(DEVICE)
    for ep in range(300):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 32):
            idx = perm[i: i + 32]
            opt.zero_grad()
            loss = crit(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()

    ROUTER_DIR.mkdir(exist_ok=True)
    weights_path = ROUTER_DIR / "router_mlp_v3.pt"
    scaler_path = ROUTER_DIR / "scaler_v3.pkl"
    torch.save(model.state_dict(), weights_path)
    with scaler_path.open("wb") as f:
        pickle.dump(final_scaler, f)
    print(f"  saved {weights_path}")
    print(f"  saved {scaler_path}")

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(Xs).to(DEVICE)).argmax(-1).cpu().numpy()
    print("\nFinal model accuracy on training set (sanity check):")
    print(f"  {(preds == y).mean() * 100:.2f}%")
    print("\nConfusion matrix (rows = true mode, cols = predicted):")
    cm = confusion_matrix(y, preds, labels=[0, 1, 2])
    print(cm)
    print("\nClassification report:")
    print(classification_report(y, preds, target_names=["Mode 1", "Mode 2", "Mode 3"]))


if __name__ == "__main__":
    main()
