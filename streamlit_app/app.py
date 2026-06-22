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
# Clinical color palette — desaturated, calm, no alarm-coded hues.
# Slate gray, medical blue, dark teal. All have AA contrast on white.
MODE_COLORS = {1: "#546E7A", 2: "#1565C0", 3: "#00695C"}


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
    # Prefer the v2 file (400 patients) when present, fall back to v1 (50 patients)
    patients_v2 = DATA_DIR / "patients_v2.jsonl"
    patients_v1 = DATA_DIR / "patients.jsonl"
    patients = _load_jsonl(patients_v2 if patients_v2.exists() else patients_v1)
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
    """EHR-style Patient Demographics Banner (PDB) modeled on Epic/Cerner.

    Always-visible top banner with name, MRN, demographics, and an
    allergy alert band. Clinical metadata below. Patient chart and
    prescriptions in collapsed expanders below the banner.
    """
    name = patient.get("name", "Unknown")
    age = patient.get("age", "—")
    gender = patient.get("gender", "—")
    pid = patient.get("patient_id", "—")
    dob = patient.get("dob", "—")
    n_meds = len(patient.get("medications", []))
    n_conds = len(patient.get("active_conditions", []))
    allergies = patient.get("allergies", [])
    n_allerg = len(allergies)

    # Allergy band: red if present, neutral if not. Lists substances inline.
    if n_allerg > 0:
        substances = ", ".join(a.get("substance", "Unknown") for a in allergies)
        allergy_html = (
            f"<div style='background:#fdecea;border-left:4px solid #c62828;"
            f"padding:6px 12px;margin-top:8px;font-size:13px;color:#7f1d1d;'>"
            f"<strong>ALLERGIES ({n_allerg})</strong> &nbsp;·&nbsp; {substances}"
            f"</div>"
        )
    else:
        allergy_html = (
            f"<div style='background:#f3f4f6;border-left:4px solid #9ca3af;"
            f"padding:6px 12px;margin-top:8px;font-size:13px;color:#374151;'>"
            f"<strong>ALLERGIES</strong> &nbsp;·&nbsp; No documented drug allergies"
            f"</div>"
        )

    banner_html = (
        f"<div style='border:1px solid #d1d5db;border-radius:6px;"
        f"padding:14px 18px;margin-bottom:8px;background:#ffffff;'>"
        # Row 1: Name + MRN
        f"<div style='display:flex;justify-content:space-between;"
        f"align-items:baseline;margin-bottom:4px;'>"
        f"<div style='font-size:22px;font-weight:700;color:#111827;'>{name}</div>"
        f"<div style='font-family:ui-monospace,Menlo,Monaco,monospace;"
        f"font-size:13px;color:#6b7280;'>MRN: {pid}</div>"
        f"</div>"
        # Row 2: Demographics
        f"<div style='font-size:13px;color:#4b5563;margin-bottom:2px;'>"
        f"<span>{age} years</span>"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;<span>Sex: {gender}</span>"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;<span>DOB: {dob}</span>"
        f"</div>"
        # Row 3: Counts
        f"<div style='font-size:12px;color:#6b7280;text-transform:uppercase;"
        f"letter-spacing:0.5px;margin-top:6px;'>"
        f"<span style='color:#374151;'><strong>{n_conds}</strong> active condition"
        f"{'s' if n_conds != 1 else ''}</span>"
        f"&nbsp;&nbsp;&nbsp;<span style='color:#374151;'>"
        f"<strong>{n_meds}</strong> prescription{'s' if n_meds != 1 else ''}</span>"
        f"</div>"
        # Allergy band
        f"{allergy_html}"
        f"</div>"
    )
    st.markdown(banner_html, unsafe_allow_html=True)

    # Collapsed: Patient history (chart sections)
    with st.expander("Patient history (conditions, allergies, vitals, lifestyle)",
                     expanded=False):
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

    # Collapsed: Active prescriptions (separate expander — this is the
    # block users most often want to see when verifying drug context)
    with st.expander(f"Active prescriptions ({n_meds})", expanded=False):
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

    def _relevance(score: float) -> str:
        """Convert cross-encoder logit to a 0–100 relevance percentage via sigmoid."""
        import math
        try:
            p = 1 / (1 + math.exp(-float(score)))
            return f"{p * 100:.0f}% rel."
        except Exception:
            return f"score {score}"

    with st.expander("retrieval / KG / MedlinePlus details"):
        lat = mode_data.get("latency_seconds", 0.0)
        n_chunks = mode_data.get("n_retrieved_chunks", 0)
        n_med = mode_data.get("n_medlineplus_chunks", 0)
        n_kg = mode_data.get("n_kg_triples", 0)
        st.caption(
            f"Latency: {lat:.2f}s · patient chunks: {n_chunks} · "
            f"MedlinePlus chunks: {n_med} · KG triples: {n_kg}"
        )
        chunks = mode_data.get("retrieved_chunks", [])
        if chunks:
            st.markdown("**Retrieved patient chunks**")
            for c in chunks:
                st.markdown(
                    f"- _{c.get('section', '?')}_ "
                    f"({_relevance(c.get('score', 0))}): "
                    f"{c.get('text', '')[:160]}…"
                )
        med_chunks = mode_data.get("medlineplus_chunks", [])
        if med_chunks:
            st.markdown("**Retrieved MedlinePlus chunks**")
            for c in med_chunks:
                title = c.get("title", "?")
                url = c.get("url", "")
                title_md = f"[{title}]({url})" if url else title
                st.markdown(
                    f"- {title_md} ({_relevance(c.get('score', 0))}): "
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
    # Compact app header — clinical software convention is a small,
    # text-only title bar with system status, not a hero section.
    mode_pill = (
        "<span style='background:#dcfce7;color:#166534;padding:2px 8px;"
        "border-radius:3px;font-size:11px;font-weight:600;letter-spacing:0.5px;"
        "text-transform:uppercase;'>LIVE</span>"
        if live_mode else
        "<span style='background:#e5e7eb;color:#374151;padding:2px 8px;"
        "border-radius:3px;font-size:11px;font-weight:600;letter-spacing:0.5px;"
        "text-transform:uppercase;'>MOCK</span>"
    )
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;"
        f"align-items:center;border-bottom:1px solid #e5e7eb;"
        f"padding-bottom:8px;margin-bottom:14px;'>"
        f"<div style='font-size:18px;font-weight:600;color:#111827;'>"
        f"MedRoute &nbsp;<span style='font-weight:400;color:#6b7280;'>"
        f"Patient Portal</span></div>"
        f"<div>{mode_pill}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

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

    render_patient_card(patient)
    st.markdown(
        "<div style='margin-top:18px;font-size:13px;color:#6b7280;"
        "text-transform:uppercase;letter-spacing:0.6px;font-weight:600;'>"
        "Patient Question</div>",
        unsafe_allow_html=True,
    )

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
    btn_cols = st.columns([3, 3, 6])
    run_router = btn_cols[0].button(
        "Run", type="primary", key=f"mock_router_{qid}",
        use_container_width=True,
        help="Use the MLP router to pick one mode and show that answer",
    )
    run_all = btn_cols[1].button(
        "Run all 3 modes", key=f"mock_all_{qid}",
        use_container_width=True,
    )

    if "mock_view" not in st.session_state:
        st.session_state.mock_view = "router"
    if run_router:
        st.session_state.mock_view = "router"
    elif run_all:
        st.session_state.mock_view = "all"

    pred = data["router_preds"].get(qid)
    response = data["responses"].get(qid)
    gt = data["ground_truth"].get(qid)
    if not response:
        st.warning("No saved response for this question.")
        return

    st.markdown("### Router decision")
    render_router_card(pred, expected=chosen["expected_mode"])

    st.markdown("---")
    if st.session_state.mock_view == "router" and pred is not None:
        picked = pred["predicted_mode"]
        st.markdown(f"### Final answer (Mode {picked} — {MODE_NAMES[picked]})")
        mode_data = response.get(f"mode_{picked}", {})
        correctness = gt.get(f"mode_{picked}_correct") if gt else None
        render_mode_card(picked, mode_data, correctness, is_router_pick=True)
    else:
        st.markdown("### Three-Mode Responses")
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
    cols = st.columns(5)
    cols[0].markdown(f"**Encoders**: {'✓' if status['encoders'] else '✗'}")
    cols[1].markdown(f"**Patient FAISS**: {'✓' if status['patient_index'] else '✗'}")
    cols[2].markdown(f"**MedlinePlus**: {'✓' if status.get('medlineplus') else '✗'}")
    cols[3].markdown(f"**KG**: {'✓' if status['kg'] else '✗'}")
    cols[4].markdown(f"**Ollama**: {'✓' if status['ollama'] else '✗'}")
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
    btn_cols = st.columns([3, 3, 6])
    run_router = btn_cols[0].button(
        "Run", type="primary", use_container_width=True,
        help="Use the MLP router to pick one mode and only call the LLM for that mode",
    )
    run_all = btn_cols[1].button("Run all 3 modes", use_container_width=True)

    if not (run_router or run_all):
        st.caption("Enter a question and press a button.")
        return
    if not question.strip():
        st.warning("Please type a question first.")
        return

    from lib.router import predict_category
    category = predict_category(question)

    if run_all:
        progress = st.progress(0.0, text="Running Mode 1 (LLM only)…")
        r1 = backend.mode_1(question, patient)
        progress.progress(0.33, text="Running Mode 2 (+ RAG)…")
        r2 = backend.mode_2(question, patient)
        progress.progress(0.66, text="Running Mode 3 (+ RAG + KG)…")
        r3 = backend.mode_3(question, patient)
        progress.progress(1.0, text="Done.")
        progress.empty()

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
        return

    # run_router path: prepare features once, route, call LLM for picked mode only
    progress = st.progress(0.0, text="Retrieval + KG lookup…")
    reranked, all_triples, m2_partial, m3_partial = backend.prepare_features(
        question, patient
    )
    progress.progress(0.5, text="Routing…")
    router_pred = router.predict(
        question=question, category=category, patient=patient,
        mode_2_record=m2_partial, mode_3_record=m3_partial,
    )
    picked = router_pred["predicted_mode"]
    progress.progress(0.7, text=f"Generating Mode {picked} answer…")
    final_record = backend.answer_for_mode(
        picked, question, patient, reranked, all_triples
    )
    progress.progress(1.0, text="Done.")
    progress.empty()

    st.markdown(f"**Question:** {question}")
    st.caption(f"Auto-detected category: `{category}`")
    st.markdown("---")
    st.markdown("### Router decision")
    render_router_card(router_pred, fallback=router_pred.get("fallback", False))
    st.markdown("---")
    st.markdown(f"### Final answer (Mode {picked} — {MODE_NAMES[picked]})")
    render_mode_card(picked, final_record, correctness=None, is_router_pick=True)


def page_overview(data: dict) -> None:
    # Compact header consistent with the portal page
    st.markdown(
        "<div style='display:flex;justify-content:space-between;"
        "align-items:center;border-bottom:1px solid #e5e7eb;"
        "padding-bottom:8px;margin-bottom:16px;'>"
        "<div style='font-size:18px;font-weight:600;color:#111827;'>"
        "MedRoute &nbsp;<span style='font-weight:400;color:#6b7280;'>"
        "System Overview</span></div>"
        "<div style='font-size:12px;color:#6b7280;text-transform:uppercase;"
        "letter-spacing:0.5px;'>CSE 676A · Spring 2026</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ---- Section: Dataset summary -----------------------------------------
    n_patients = len(data["patients"])
    n_train = sum(1 for v in data["split_label"].values() if v == "train")
    n_test = sum(1 for v in data["split_label"].values() if v == "test")

    _section_label("Dataset")
    cols = st.columns(3)
    _metric_box(cols[0], "Synthetic patients", n_patients,
                 "FHIR-compliant records")
    _metric_box(cols[1], "Training questions", n_train,
                 "Used to fit MLP router")
    _metric_box(cols[2], "Held-out test questions", n_test,
                 "Stratified by category")

    # ---- Section: Architecture --------------------------------------------
    _section_label("System architecture")
    st.graphviz_chart(_architecture_dot(), use_container_width=True)

    st.caption(
        "Patient question + chart → MLP router selects one of three answer "
        "pipelines based on retrieval, knowledge-graph, and patient-state "
        "features. Each mode runs only the components it needs."
    )

    # ---- Section: Knowledge sources ---------------------------------------
    _section_label("Knowledge sources")
    st.markdown(
        """
| Source | Purpose | Coverage |
|--------|---------|----------|
| **Patient FAISS** | Per-patient EHR chunks (demographics, conditions, allergies, vitals, lifestyle) | 400 patients · 1,633 chunks |
| **MedlinePlus FAISS** | NIH consumer-health articles (drugs, conditions, procedures) | 2,127 articles · 15,803 chunks |
| **PrimeKG** | Structured biomedical knowledge graph (drug-drug, drug-disease, drug-side-effect) | 128,550 entities · 4M relations |
| **MedCPT encoders** | Query / Article / Cross-Encoder for biomedical retrieval (NCBI, NIH) | 110M params · 64-token query · 512-token cross |
| **Gemma2 9B** | Local generative model via Ollama (4-bit quantized, no PHI egress) | ~5 GB VRAM |
        """
    )

    # ---- Section: Router architecture -------------------------------------
    _section_label("MLP Router (27 features)")
    cols = st.columns([1, 1])
    with cols[0]:
        st.markdown(
            """
**Network**

`Linear(27 → 32) → ReLU → Dropout(0.25)`
`Linear(32 → 16) → ReLU → Dropout(0.25)`
`Linear(16 → 3) → Softmax`

1,507 trainable parameters. Trained with class-weighted cross-entropy
(balanced inverse frequency) on 5-fold stratified CV.
            """
        )
    with cols[1]:
        st.markdown(
            """
**Features (27)**

| Block | Dim |
|-------|----:|
| Question type one-hot | 12 |
| Patient retrieval scores | 4 |
| MedlinePlus retrieval scores | 4 |
| KG features | 2 |
| Patient features | 3 |
| Question features | 2 |
            """
        )

    # ---- Section: Results -------------------------------------------------
    _section_label("Results — strict medically-aware evaluation (n=2000)")
    st.markdown(
        """
| Strategy | Accuracy | Per-question compute |
|----------|---------:|---------------------:|
| Always Mode 1 (LLM only) | 40.2% | 1.0× |
| Always Mode 2 (+RAG) | 95.2% | 2.0× |
| Always Mode 3 (+RAG+KG) | **96.6%** | 3.0× |
| **MLP Router** | **~96%** end-user | **~1.74×** (40% M1 + 43% M2 + 17% M3) |
| Oracle (best of 3) | 99.3% | 3.0× |

Routing achieves end-user accuracy within 0.5pp of the strongest fixed
mode while reducing per-question compute by ~42%. Mode 3 recall = 99% —
zero safety-critical questions misrouted to LLM-only.
        """
    )

    _section_label("Safety guarantees")
    st.markdown(
        """
- **Mode 1 disqualified on safety questions** — drug interactions, food interactions, lifestyle safety, side effects all require grounded sources (RAG or KG). LLM parametric knowledge alone is not trusted.
- **Citation verification** — every `[N]` citation in Mode 2 / Mode 3 answers is scored against its source via the MedCPT cross-encoder; low-relevance citations are dropped automatically.
- **No PHI egress** — Gemma2 runs locally via Ollama. Patient data never leaves the host.
        """
    )


# ---- Overview helpers ------------------------------------------------------

def _section_label(text: str) -> None:
    st.markdown(
        f"<div style='font-size:11px;color:#6b7280;text-transform:uppercase;"
        f"letter-spacing:0.6px;font-weight:600;margin-top:22px;"
        f"margin-bottom:8px;border-bottom:1px solid #e5e7eb;"
        f"padding-bottom:4px;'>{text}</div>",
        unsafe_allow_html=True,
    )


def _metric_box(col, label: str, value, sublabel: str) -> None:
    col.markdown(
        f"<div style='border:1px solid #e5e7eb;border-radius:6px;"
        f"padding:14px;background:#fafafa;'>"
        f"<div style='font-size:11px;color:#6b7280;text-transform:uppercase;"
        f"letter-spacing:0.4px;font-weight:600;'>{label}</div>"
        f"<div style='font-size:24px;font-weight:700;color:#111827;"
        f"margin-top:4px;'>{value}</div>"
        f"<div style='font-size:12px;color:#6b7280;margin-top:2px;'>"
        f"{sublabel}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _architecture_dot() -> str:
    """Graphviz DOT for the system architecture diagram.
    Clinical color scheme — desaturated blues, grays, teals."""
    return r"""
    digraph MedRoute {
        rankdir=TB;
        bgcolor="white";
        node [fontname="Helvetica", fontsize=11, style="filled,rounded",
              shape=box, color="#9ca3af", penwidth=1.0];
        edge [fontname="Helvetica", fontsize=9, color="#6b7280",
              arrowsize=0.7];

        // Inputs
        Q [label="Patient question", fillcolor="#e5e7eb", color="#9ca3af"];
        Pat [label="Patient EHR\n(demographics, vitals,\nconditions, allergies)",
             fillcolor="#e5e7eb", color="#9ca3af"];

        // Feature extraction
        Feat [label="Feature extraction\n27 features",
              fillcolor="#f3f4f6", color="#9ca3af"];

        // Router
        Rt [label="MLP Router\n27 → 32 → 16 → 3",
            fillcolor="#fff8e1", color="#d97706", fontcolor="#92400e"];

        // Modes
        M1 [label="Mode 1\nLLM only\n(Gemma2-9B)",
            fillcolor="#eceff1", color="#546e7a", fontcolor="#37474f"];
        M2 [label="Mode 2\nLLM + Patient RAG\n+ MedlinePlus + citations",
            fillcolor="#e3f2fd", color="#1565c0", fontcolor="#0d47a1"];
        M3 [label="Mode 3\nLLM + Patient RAG\n+ KG + MedlinePlus",
            fillcolor="#e0f2f1", color="#00695c", fontcolor="#004d40"];

        // Knowledge sources (cluster)
        subgraph cluster_kb {
            label="Knowledge bases"; fontsize=10; color="#d1d5db";
            style="rounded,dashed";
            Patf [label="Patient FAISS\n(MedCPT)", fillcolor="#fafafa", color="#9ca3af"];
            MedF [label="MedlinePlus FAISS\n2,127 articles", fillcolor="#fafafa", color="#9ca3af"];
            KG [label="PrimeKG\n4M relations", fillcolor="#fafafa", color="#9ca3af"];
        }

        // Final answer
        Ans [label="Cited answer\nto patient", fillcolor="#fef3c7", color="#a16207"];

        // Edges
        Q -> Feat;
        Pat -> Feat;
        Feat -> Rt;
        Rt -> M1 [label="P(M1)"];
        Rt -> M2 [label="P(M2)"];
        Rt -> M3 [label="P(M3)"];

        M2 -> Patf [style=dashed];
        M2 -> MedF [style=dashed];
        M3 -> Patf [style=dashed];
        M3 -> MedF [style=dashed];
        M3 -> KG [style=dashed];

        M1 -> Ans;
        M2 -> Ans;
        M3 -> Ans;
    }
    """


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="MedRoute Patient Portal",
        page_icon=None,
        layout="wide",
    )

    # Inject clinical-style CSS overrides — hospital software aesthetic.
    # Targets components that .streamlit/config.toml doesn't reach: button
    # accents, expander chrome, sidebar borders, tighter typography.
    st.markdown(
        """
        <style>
        /* Body font + tighter headings */
        html, body, [class*="css"] {
            font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue",
                         Arial, sans-serif;
            color: #0f172a;
        }
        h1, h2, h3 { font-weight: 600; letter-spacing: -0.01em; }

        /* Page background — very light clinical blue (hospital ambient) */
        .stApp { background-color: #eff6fb; }

        /* Sidebar — slightly deeper blue-gray with right border */
        section[data-testid="stSidebar"] {
            background-color: #e2ecf5;
            border-right: 1px solid #cbd5e1;
        }
        section[data-testid="stSidebar"] * { color: #1e293b; }

        /* Primary buttons — medical blue, no aggressive red */
        .stButton > button[kind="primary"],
        button[data-testid="baseButton-primary"] {
            background-color: #1565c0 !important;
            border: 1px solid #1565c0 !important;
            color: #ffffff !important;
            font-weight: 600;
            border-radius: 4px;
            padding: 8px 16px;
            transition: background-color 0.15s ease;
        }
        .stButton > button[kind="primary"]:hover,
        button[data-testid="baseButton-primary"]:hover {
            background-color: #0d47a1 !important;
            border-color: #0d47a1 !important;
        }

        /* Secondary / default buttons — outline blue */
        .stButton > button:not([kind="primary"]) {
            background-color: #ffffff;
            border: 1px solid #cbd5e1;
            color: #1e293b;
            font-weight: 500;
            border-radius: 4px;
        }
        .stButton > button:not([kind="primary"]):hover {
            border-color: #1565c0;
            color: #1565c0;
            background-color: #f1f5f9;
        }

        /* Toggle / switch — replace red with clinical blue */
        div[data-testid="stToggle"] label > div[data-baseweb="checkbox"] > div {
            background-color: #1565c0 !important;
        }

        /* Expanders — flatter, hospital-doc style */
        details[data-testid="stExpander"] {
            border: 1px solid #e2e8f0 !important;
            border-radius: 4px !important;
            background-color: #ffffff;
        }
        details[data-testid="stExpander"] summary {
            font-weight: 500;
            color: #334155;
            background-color: #f8fafc;
            border-radius: 4px;
        }

        /* Text inputs — calmer borders */
        input[data-testid="stTextInput"], .stTextInput > div > div > input {
            border: 1px solid #cbd5e1 !important;
            border-radius: 4px !important;
        }
        input[data-testid="stTextInput"]:focus,
        .stTextInput > div > div > input:focus {
            border-color: #1565c0 !important;
            box-shadow: 0 0 0 1px #1565c0 !important;
        }

        /* Code / monospace blocks — softer */
        code { background-color: #f1f5f9; color: #0f172a;
               padding: 1px 6px; border-radius: 3px; font-size: 12px; }

        /* Hide only the Streamlit footer + deploy/hamburger menu.
           DO NOT hide the header — it contains the sidebar toggle. */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        /* Make header transparent but keep it interactive */
        header[data-testid="stHeader"] { background: transparent; }

        /* Tighter section spacing */
        .block-container { padding-top: 1.5rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
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
        "Toggle live mode to use real LLM."
    )

    if page == "Patient Portal":
        page_portal(data, live_mode=live_mode)
    else:
        page_overview(data)


if __name__ == "__main__":
    main()
