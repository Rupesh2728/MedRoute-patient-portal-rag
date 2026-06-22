"""
Hybrid RAG Patient Portal — Streamlit Demo (live + mock modes).

Two modes:
  - Mock mode: shows pre-computed responses from the JSONL files. No
    Ollama or model loads needed; works on any laptop.
  - Live mode: calls the real pipeline (MedCPT retrieval, PrimeKG lookup,
    Gemma2 via Ollama) for each question typed by the user.

Run from this folder:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).parent
PROJECT_DIR = APP_DIR.parent
DATA_DIR = PROJECT_DIR / "synthetic_patients"
ROUTER_DIR = PROJECT_DIR / "patient_router"
TESTSET_DIR = PROJECT_DIR / "patient_portal_testset"

MODE_NAMES = {1: "LLM only", 2: "+ RAG", 3: "+ RAG + KG"}
MODE_COLORS = {1: "#264653", 2: "#2A9D8F", 3: "#E76F51"}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


@st.cache_data
def load_mock_data() -> dict[str, Any]:
    patients = _load_jsonl(DATA_DIR / "patients.jsonl")
    train_q = _load_jsonl(DATA_DIR / "questions.jsonl")
    test_q = _load_jsonl(DATA_DIR / "test_questions.jsonl")

    train_resp_path = DATA_DIR / "patient_portal_responses_v2.jsonl"
    if not train_resp_path.exists():
        train_resp_path = DATA_DIR / "patient_portal_responses.jsonl"
    train_resp = _load_jsonl(train_resp_path)
    test_resp = _load_jsonl(DATA_DIR / "test_responses.jsonl")

    train_gt_path = DATA_DIR / "patient_portal_ground_truth_v2.jsonl"
    if not train_gt_path.exists():
        train_gt_path = DATA_DIR / "patient_portal_ground_truth.jsonl"
    train_gt = _load_jsonl(train_gt_path)
    test_gt = _load_jsonl(DATA_DIR / "test_ground_truth.jsonl")

    train_router_preds = _load_jsonl(ROUTER_DIR / "router_predictions.jsonl")
    test_router_preds = _load_jsonl(TESTSET_DIR / "test_router_predictions.jsonl")

    return {
        "patients": {p["patient_id"]: p for p in patients},
        "questions": {q["question_id"]: q for q in (train_q + test_q)},
        "responses": {r["question_id"]: r for r in (train_resp + test_resp)},
        "ground_truth": {g["question_id"]: g for g in (train_gt + test_gt)},
        "router_preds": {p["question_id"]: p for p in (train_router_preds + test_router_preds)},
        "questions_by_patient": _group_by_patient(train_q + test_q),
        "split_label": _build_split_label(train_q, test_q),
    }


def _group_by_patient(questions: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for q in questions:
        out.setdefault(q["patient_id"], []).append(q)
    return out


def _build_split_label(train_q: list[dict], test_q: list[dict]) -> dict[str, str]:
    label = {}
    for q in train_q:
        label[q["question_id"]] = "train"
    for q in test_q:
        label[q["question_id"]] = "test"
    return label


@st.cache_resource(show_spinner="Loading models (first call only — ~30s)…")
def load_live_backend():
    """Lazy-loaded; only constructed when user enables Live mode."""
    from lib.pipeline import LiveBackend
    from lib.router import Router

    backend = LiveBackend(project_dir=PROJECT_DIR)
    backend.load()
    router = Router(project_dir=PROJECT_DIR)
    router.load()
    return backend, router


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def render_patient_card(patient: dict) -> None:
    name = patient.get("name", "Unknown")
    age = patient.get("age", "—")
    gender = patient.get("gender", "—")
    pid = patient.get("patient_id", "—")

    st.markdown(
        f"### {name}  \n_{age}-year-old {gender.lower()} • {pid}_"
    )

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Active Conditions**")
        conds = patient.get("active_conditions", [])
        if conds:
            for c in conds:
                st.markdown(f"- {c['condition']} _(diagnosed {c['diagnosed']})_")
        else:
            st.caption("None documented")

        st.markdown("**Allergies**")
        allergies = patient.get("allergies", [])
        if allergies:
            for a in allergies:
                st.markdown(f"- {a['substance']}: {a.get('reaction', 'unspecified')}")
        else:
            st.caption("No documented drug allergies")

    with cols[1]:
        st.markdown("**Recent Vitals**")
        vitals = patient.get("recent_vitals", {})
        if vitals:
            date = vitals.get("date", "")
            for k, v in vitals.items():
                if k == "date":
                    continue
                st.markdown(f"- **{k.replace('_', ' ').title()}**: {v}")
            if date:
                st.caption(f"_(as of {date})_")
        else:
            st.caption("No recent vitals")

        st.markdown("**Lifestyle**")
        ls = patient.get("lifestyle", {})
        if ls:
            for k, v in ls.items():
                st.markdown(f"- **{k.title()}**: {v}")
        else:
            st.caption("No lifestyle info")

    st.markdown("---")
    st.markdown("### Active Prescriptions")
    meds = patient.get("medications", [])
    if not meds:
        st.info("No active prescriptions on file.")
    for m in meds:
        with st.container(border=True):
            st.markdown(f"**{m['drug']}** — {m['dosage']}, {m['frequency']}")
            st.markdown(f"_For:_ {m['indication']}")
            if m.get("notes"):
                st.markdown(f"_Notes:_ {m['notes']}")


def render_mode_card(
    mode_num: int,
    mode_data: dict,
    correctness: int | None = None,
    is_router_pick: bool = False,
) -> None:
    color = MODE_COLORS[mode_num]
    label = MODE_NAMES[mode_num]
    border_label = "ROUTER PICK" if is_router_pick else ""
    header = (
        f"<div style='border-left: 4px solid {color}; padding-left: 8px;'>"
        f"<div style='font-size: 12px; color: #6c7a89; letter-spacing: 1px;'>"
        f"MODE {mode_num} {border_label}</div>"
        f"<div style='font-size: 18px; font-weight: 600; color: {color};'>{label}</div>"
        f"</div>"
    )
    st.markdown(header, unsafe_allow_html=True)

    answer = (mode_data.get("answer") or "").strip()
    if not answer:
        st.warning("No response generated.")
        return

    if correctness == 1:
        st.markdown(
            "<div style='color:#2A9D8F;font-weight:600;margin:6px 0;'>✓ scored correct</div>",
            unsafe_allow_html=True,
        )
    elif correctness == 0:
        st.markdown(
            "<div style='color:#E76F51;font-weight:600;margin:6px 0;'>✗ scored incorrect</div>",
            unsafe_allow_html=True,
        )

    st.markdown(answer)

    with st.expander("retrieval / KG details"):
        lat = mode_data.get("latency_seconds", 0.0)
        n_chunks = mode_data.get("n_retrieved_chunks", 0)
        n_kg = mode_data.get("n_kg_triples", 0)
        st.caption(f"Latency: {lat:.2f}s · chunks: {n_chunks} · KG triples: {n_kg}")
        chunks = mode_data.get("retrieved_chunks", [])
        if chunks:
            st.markdown("**Retrieved patient chunks**")
            for c in chunks:
                st.markdown(
                    f"- _{c.get('section', '?')}_ "
                    f"(score {c.get('score', 0):.2f}): "
                    f"{c.get('text', '')[:160]}…"
                )
        kg_triples = mode_data.get("kg_triples", [])
        if kg_triples:
            st.markdown("**KG triples**")
            for t in kg_triples[:8]:
                st.markdown(f"- {t}")
            if len(kg_triples) > 8:
                st.caption(f"…and {len(kg_triples) - 8} more")


def render_router_card(
    pred: dict | None,
    expected: int | None = None,
    fallback: bool = False,
) -> None:
    if not pred:
        st.info("No router prediction available.")
        return

    pred_mode = pred["predicted_mode"]
    confidence = pred.get("confidence", 0.0)
    probs = pred.get("probs", [0.0, 0.0, 0.0])
    correct = (
        pred.get("correct")
        if "correct" in pred
        else (expected is not None and pred_mode == expected)
    )

    cols = st.columns([1, 1, 1])
    with cols[0]:
        st.metric(
            "Router predicts",
            f"Mode {pred_mode} ({MODE_NAMES[pred_mode]})",
            f"{confidence * 100:.1f}% confidence",
        )
    with cols[1]:
        if expected is not None:
            st.metric(
                "Expected mode",
                f"Mode {expected} ({MODE_NAMES[expected]})",
                "✓ match" if correct else "✗ mismatch",
                delta_color="normal" if correct else "inverse",
            )
        else:
            st.metric("Expected mode", "—", "(no label for free-text question)",
                       delta_color="off")
    with cols[2]:
        st.metric(
            "Mode probabilities",
            f"{probs[0]:.2f} / {probs[1]:.2f} / {probs[2]:.2f}",
            "M1 / M2 / M3",
            delta_color="off",
        )

    if fallback:
        st.warning(
            "Router model not loaded — defaulting to Mode 2. "
            "Place `router_mlp.pt` and `scaler.pkl` in `patient_router/` to enable."
        )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_portal(data: dict, live_mode: bool) -> None:
    st.title("Hybrid RAG Patient Portal")
    if live_mode:
        st.caption("LIVE: each question is answered by Gemma2 via Ollama in real time.")
    else:
        st.caption("MOCK: showing pre-computed responses from the saved JSONL files.")

    patients = data["patients"]
    if not patients:
        st.error(f"No patients found at `{DATA_DIR.relative_to(PROJECT_DIR)}/`.")
        return

    sorted_patients = sorted(patients.values(), key=lambda p: p["patient_id"])
    patient_options = [f"{p['patient_id']} — {p['name']}" for p in sorted_patients]
    sel = st.sidebar.selectbox(
        "Select patient",
        list(range(len(patient_options))),
        format_func=lambda i: patient_options[i],
    )
    patient = sorted_patients[sel]

    n_q = len(data["questions_by_patient"].get(patient["patient_id"], []))
    st.sidebar.markdown("---")
    st.sidebar.metric("Questions on file", n_q)
    st.sidebar.metric("Active prescriptions", len(patient.get("medications", [])))

    render_patient_card(patient)
    st.markdown("---")
    st.markdown("### Ask a Question")

    if live_mode:
        _ask_question_live(patient)
    else:
        _ask_question_mock(patient, data)


def _ask_question_mock(patient: dict, data: dict) -> None:
    qs_for_patient = data["questions_by_patient"].get(patient["patient_id"], [])
    if not qs_for_patient:
        st.info("No pre-computed questions for this patient.")
        return

    options = [
        f"[{data['split_label'].get(q['question_id'], '?').upper()}] "
        f"{q['question_id']}: {q['question'][:90]}"
        for q in qs_for_patient
    ]
    q_idx = st.selectbox(
        "Pick a pre-computed question",
        list(range(len(qs_for_patient))),
        format_func=lambda i: options[i],
    )
    chosen = qs_for_patient[q_idx]
    qid = chosen["question_id"]

    st.markdown(f"**Question:** {chosen['question']}")
    st.caption(
        f"Category: `{chosen['category']}` • "
        f"Expected mode: **Mode {chosen['expected_mode']}** "
        f"({chosen['expected_mode_name']})"
    )
    if chosen.get("rationale"):
        st.caption(f"Rationale: _{chosen['rationale']}_")

    st.markdown("---")
    st.markdown("### Router decision")
    pred = data["router_preds"].get(qid)
    render_router_card(pred, expected=chosen["expected_mode"])

    st.markdown("---")
    st.markdown("### Three-Mode Responses")
    response = data["responses"].get(qid)
    gt = data["ground_truth"].get(qid)
    if not response:
        st.warning("No saved response for this question.")
        return

    cols = st.columns(3)
    for i, m in enumerate([1, 2, 3]):
        with cols[i]:
            mode_data = response.get(f"mode_{m}", {})
            correctness = gt.get(f"mode_{m}_correct") if gt else None
            is_router_pick = pred is not None and pred.get("predicted_mode") == m
            render_mode_card(m, mode_data, correctness, is_router_pick)


def _ask_question_live(patient: dict) -> None:
    backend, router = load_live_backend()
    status = backend.status_summary()

    # Status banner
    cols = st.columns(4)
    cols[0].markdown(f"**Encoders**: {'✓' if status['encoders'] else '✗'}")
    cols[1].markdown(f"**Patient FAISS**: {'✓' if status['patient_index'] else '✗'}")
    cols[2].markdown(f"**KG**: {'✓' if status['kg'] else '✗'}")
    cols[3].markdown(f"**Ollama**: {'✓' if status['ollama'] else '✗'}")
    if backend.warnings:
        with st.expander("Loader warnings"):
            for w in backend.warnings:
                st.caption(f"- {w}")
    if not status["ollama"]:
        st.error("Ollama not reachable. Start it with `ollama serve` and ensure `gemma2` is pulled.")
        return

    question = st.text_input(
        "Type your question",
        placeholder="e.g., Can I take ibuprofen with my prescriptions?",
    )
    run_btn = st.button("Run all 3 modes", type="primary")

    if not run_btn or not question.strip():
        st.caption("Enter a question and press the button.")
        return

    # Run pipeline
    progress = st.progress(0.0, text="Running Mode 1 (LLM only)…")
    r1 = backend.mode_1(question, patient)
    progress.progress(0.33, text="Running Mode 2 (+ RAG)…")
    r2 = backend.mode_2(question, patient)
    progress.progress(0.66, text="Running Mode 3 (+ RAG + KG)…")
    r3 = backend.mode_3(question, patient)
    progress.progress(1.0, text="Done.")
    progress.empty()

    # Router prediction (uses the real retrieval/KG outputs we just computed)
    from lib.router import predict_category
    category = predict_category(question)
    router_pred = router.predict(
        question=question, category=category, patient=patient,
        mode_2_record=r2, mode_3_record=r3,
    )

    st.markdown(f"**Question:** {question}")
    st.caption(f"Auto-detected category: `{category}`")

    st.markdown("---")
    st.markdown("### Router decision")
    render_router_card(router_pred, fallback=router_pred.get("fallback", False))

    st.markdown("---")
    st.markdown("### Three-Mode Responses")
    cols = st.columns(3)
    for i, (m, rec) in enumerate(zip([1, 2, 3], [r1, r2, r3])):
        with cols[i]:
            is_router_pick = router_pred and router_pred["predicted_mode"] == m
            render_mode_card(m, rec, correctness=None, is_router_pick=is_router_pick)


def page_overview(data: dict) -> None:
    st.title("Project Overview")
    st.caption("Hybrid RAG with adaptive routing for medical Q&A")

    n_patients = len(data["patients"])
    n_train = sum(1 for v in data["split_label"].values() if v == "train")
    n_test = sum(1 for v in data["split_label"].values() if v == "test")

    cols = st.columns(3)
    cols[0].metric("Synthetic patients", n_patients)
    cols[1].metric("Training questions", n_train)
    cols[2].metric("Held-out test questions", n_test)

    st.markdown("---")
    st.markdown("### Architecture")
    st.code(
        """Question → MLP Router → pick Mode 1 / 2 / 3
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
     Mode 1          Mode 2          Mode 3
     LLM only        + Patient RAG   + Patient RAG
     (Gemma2 + CoT)  (FAISS over     + PrimeKG triples
                      patient chunks) (drug interactions)""",
        language="text",
    )

    st.markdown("### Key results")
    st.markdown(
        """
| Strategy            | Train (250) | Test (100) |
|---------------------|-------------|------------|
| Always Mode 1       | 63.6%       | 53.0%      |
| Always Mode 2       | 92.8%       | 95.0%      |
| Always Mode 3       | 95.2%       | 97.0%      |
| **Router (MLP)**    | **94.0%**   | **92.0%**  |
| Oracle              | 97.6%       | 98.0%      |

- **Mode 3** is the strongest single mode after the prompt-fix to encourage KG use.
- **Router** trades ~2–5 pp accuracy for ~33% lower compute per question.
- **Oracle** ≈ 98% — only ~2% of questions stump all three modes.
        """
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Hybrid RAG Patient Portal",
        page_icon="🩺",
        layout="wide",
    )

    data = load_mock_data()

    page = st.sidebar.radio("Navigation", ["Patient Portal", "Project Overview"])
    st.sidebar.markdown("---")
    live_mode = st.sidebar.toggle(
        "Live mode (use real LLM)",
        value=False,
        help=(
            "When ON, the app loads MedCPT/PrimeKG/scispaCy and calls Gemma2 via "
            "Ollama for each question. When OFF, it shows pre-computed responses "
            "from the JSONL files."
        ),
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "CSE 676A · Spring 2026  \n"
        "Synthetic data only.  \n"
        "Toggle live mode to use real LLM."
    )

    if page == "Patient Portal":
        page_portal(data, live_mode=live_mode)
    else:
        page_overview(data)


if __name__ == "__main__":
    main()
