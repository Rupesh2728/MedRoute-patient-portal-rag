# Hybrid RAG + Knowledge Graph with Adaptive MLP Routing for Medical Q&A

CSE 676A — Deep Learning, Spring 2026
University at Buffalo

## Overview

We set out to build a medical question-answering system that can decide, per
question, whether the right answer comes from (a) the LLM's parametric
knowledge, (b) retrieved passages, or (c) a curated medical knowledge graph.
The work spans two datasets and produced one clear failure, several
intermediate wins, and one deployable system.

**Headline numbers:**

- **MedQA (Gemma2-9B, n=200):** best fixed mode = 63.5% (LLM + RAG + KG).
  Best learned router = 60.5%. The MLP router *underperformed* the best
  fixed mode by 3 points.
- **Patient Portal (Gemma2-9B, n=100 test):** best fixed mode = 97.0%
  (LLM + RAG + KG). Learned router = 92.0%. Oracle = 98.0%. The same router
  architecture worked here, trading 5 points of accuracy for the
  ability to skip RAG/KG on questions that don't need them.

The contrast between these two outcomes is the substance of the project.

---

## Part 1 — MedQA: building the foundation

### 1.1 The original baseline failed

Notebook: `RAG_MedQA_Gemma2_9b_MEDCPT.ipynb`
Dataset: MedQA-USMLE validation split, n=200.

| Condition | Accuracy | F1 |
|-----------|---------:|---:|
| Gemma2 alone (no retrieval) | 54.50% | 60.62% |
| Baseline RAG (MedCPT + S-PubMed) | 51.00% | 58.29% |

Adding retrieval *hurt* performance. The baseline pipeline retrieved short
passages with MedCPT, dumped them into the prompt, and let the LLM sort it
out. The retrieved passages were often loosely related to the question stem,
and the LLM was distracted by them.

This failure is the motivation for everything that follows.

### 1.2 Six engineering fixes recovered RAG

Notebook: `RAG_MedQA_Gemma2_9b_MEDCPT_upgraded.ipynb`

We added six fixes to the RAG pipeline:

1. **Stem extraction** — strip the question's preamble before encoding.
2. **Multi-query retrieval** — encode both the stem and each option, retrieve
   from both.
3. **Cross-encoder reranking** — rerank top-K candidates with the MedCPT
   Cross-Encoder before injecting.
4. **Adaptive context** — inject context only when cross-encoder confidence
   exceeds a threshold (default = 2.0).
5. **Chain-of-thought prompting** — instruct the LLM to reason step by step.
6. **Structured context** — format passages with explicit delimiters and
   labels.

Result on the same n=200 split:

| Condition | Accuracy | F1 |
|-----------|---------:|---:|
| Gemma2 alone (letter-only) | 55.00% | 60.73% |
| Gemma2 alone (CoT) | 54.50% | 60.23% |
| Old RAG | 51.00% | 58.29% |
| **Upgraded RAG (no KG)** | **57.00%** | **63.33%** |

The upgraded RAG beat both the old RAG and the LLM-alone runs. Δ = +6.0pp
over old RAG. Cross-encoder gating was the most valuable fix: it used context
on 73.5% of questions (with avg cross-encoder relevance score 5.36, well
above the 2.0 threshold).

### 1.3 PrimeKG infrastructure was solid

Notebook: `PrimeKG_KG_SmokeTest.ipynb`

Before adding KG to the pipeline, we tested whether PrimeKG actually has
entries for the entities in MedQA questions.

| Metric | Value |
|--------|------:|
| Questions tested | 200 |
| Questions with at least one PrimeKG hit | **175 (87.5%)** |

87.5% coverage is enough to make a KG-augmented condition meaningful.

### 1.4 Naive KG hurt; smart KG helped

Two notebooks tested progressively smarter ways to inject KG.

**Naive KG (`Cond4_vs_Cond5_KG.ipynb`)** — inject every question that has at
least one entity match.

| Condition | Accuracy | F1 | KG inject rate | Avg triples |
|-----------|---------:|---:|---------------:|------------:|
| Cond 4 (no KG) | 53.50% | 60.64% | 0% | 0.00 |
| Cond 5 v1 (KG, naive) | 54.00% | 61.30% | 87.5% | 4.34 |

KG bought only +0.5pp accuracy. It rescued 13 questions but broke 12. Net +1.

**Smart KG (`Cond5_v2.ipynb`)** — three changes:

1. Cross-source reranking: combine retrieved passages and KG triples into one
   candidate set, rerank everything together with the cross-encoder.
2. Natural-language formatting: convert raw `(subject, predicate, object)`
   tuples into prose ("Metformin interacts with ibuprofen").
3. Adaptive injection: only inject KG when cross-encoder relevance for the
   top KG candidate exceeds the threshold.

| Condition | Accuracy | F1 | KG inject rate |
|-----------|---------:|---:|---------------:|
| Cond 4 (no KG) | 52.50% | 59.28% | 0.0% |
| Cond 5 v1 (naive) | 53.00% | 60.89% | 87.5% |
| **Cond 5 v2 (smart)** | **56.50%** | **63.24%** | **9.5%** |

The most surprising number is the inject rate. v2 fires on only 9.5% of
questions, yet beats v1 (which fires on 87.5%) by +3.5pp. Adding fewer, more
relevant KG facts is better than adding many irrelevant ones. v2 rescued 15
questions and broke 7. Net +8.

### 1.5 MedQA ablation summary

Comparing all three modes on the same n=200 evaluation (from
`MLP_Router_Training_v2.ipynb`, where each mode was scored on every
question):

| Mode | Accuracy |
|------|---------:|
| LLM only (Gemma2-9B) | 59.5% |
| LLM + RAG (upgraded) | 61.5% |
| LLM + RAG + KG (smart) | 63.5% |
| Oracle (best of 3 per question) | 80.5% |

The 17pp gap between LLM+RAG+KG (63.5%) and Oracle (80.5%) is the headroom a
perfect router could capture.

---

## Part 2 — The MLP router: a clean failure on MedQA

### 2.1 Hypothesis

If different modes win on different questions, a learned router that takes a
question and predicts the best mode should beat any fixed mode. The Oracle's
+17pp ceiling on MedQA suggested real headroom.

### 2.2 Three router experiments, three failures

We trained the router three times. In every case it underperformed the best
fixed mode.

| Notebook | LLM | Features | Best fixed mode | Router (CV) | Δ vs best fixed |
|----------|-----|---------:|----------------:|------------:|----------------:|
| `MLP_Router_Training.ipynb` | Gemma2-9B | 12 | 64.0% (RAG+KG) | 62.5% | **−1.5pp** |
| `MLP_Router_Training_v2.ipynb` | Gemma2-9B | 23 | 63.5% (RAG+KG) | 60.5% | **−3.0pp** |
| `MLP_Router_Training_v2_2B.ipynb` | Gemma2-2B | 23 | 60.5% (RAG+KG) | 55.5% | **−5.0pp** |

Adding more features (12 → 23) did not help. Switching to a smaller LLM
(Gemma2-2B) made the gap larger, not smaller, even though the smaller model
benefits more from RAG (+10pp).

### 2.3 Why the router collapsed

We profiled the mode-overlap structure on the 500-question training set:

| Pattern | Fraction |
|---------|---------:|
| All 3 modes wrong | 28.4% |
| All 3 modes correct | 28.8% |
| Mixed (some right, some wrong) | 42.8% |

Only 43% of questions had any signal for the router to learn from. Among
those, the 23 features (question type, retrieval scores, KG triple counts,
question length) couldn't reliably tell which mode would win. The LLM's
parametric knowledge already gets ~60% right with no help, and that
correctness is largely orthogonal to the router's features.

### 2.4 Cross-domain demo: right behavior, no end-to-end gain

Notebook: `Cross_Domain_Router_Demo.ipynb`. We mixed 100 MedQA medical
questions with 100 MMLU non-medical questions to test whether the router
could at least learn to route by domain.

| Mode | All (n=200) | Medical (n=100) | Non-medical (n=100) |
|------|------------:|----------------:|--------------------:|
| LLM only | 66.0% | 52% | 80% |
| LLM + RAG | 64.0% | 70% | 58% |
| LLM + RAG + KG | 68.0% | 76% | 60% |
| Router (learned) | 67.5% | 72% | 63% |
| Oracle | 88.0% | 90% | 86% |

Router mode usage by domain:

- Medical questions: 15% routed to LLM-only, 38% to RAG, 47% to RAG+KG.
- Non-medical: 85% routed to LLM-only, 10% to RAG, 5% to RAG+KG.

The router learned the *correct* behavior (medical → RAG+KG, non-medical →
LLM-only), but the end-to-end gain over the best fixed mode was under 1pp,
because the best fixed mode was already strong on its dominant domain. The
router was right but didn't matter.

---

## Part 3 — Pivot to the patient portal

The MedQA failure pointed to one clear cause: questions where mode-correctness
is mostly orthogonal to question features. We needed a task where mode
correctness depends on patient-specific context the router can see.

### 3.1 Synthetic patient dataset

We built a 50-patient cohort (`patients.jsonl`) with realistic medical
records. Each patient has:

- Demographics (age, gender, name)
- Active conditions with diagnosis dates
- Past medical history
- Active prescriptions (drug, dosage, frequency, indication, notes)
- Allergies
- Recent vitals
- Lifestyle factors

The schema mirrors a FHIR Patient bundle (the format Synthea produces).
Cohort statistics:

- Median 2 active conditions, median 2 prescriptions
- 16% have at least one documented allergy
- Top conditions: hypertension (15), type-2 diabetes (6), hyperlipidemia (4),
  major depressive disorder (3), generalized anxiety disorder (3)
- Top drugs: lisinopril (10), metformin (6), atorvastatin (4), metoprolol
  (4), albuterol (4)

We then wrote 250 training and 100 held-out test questions across 12
categories: 6 drug-centric (general info, mechanism, interactions, food,
lifestyle, side effects) and 6 patient-centric (indication, timing, vitals,
lifestyle, side effects, monitoring). Each question has a hand-labelled
expected mode and a one-line rationale.

### 3.2 Three-mode pipeline (the patient version)

- **Mode 1 — LLM only.** Gemma2-9B sees the patient's prescription list and
  the question.
- **Mode 2 — + Patient RAG.** MedCPT query encoder + FAISS over per-patient
  chunked records. Cross-encoder reranks top-4 → top-3 chunks, which are
  injected into the prompt.
- **Mode 3 — + Patient RAG + KG.** PrimeKG lookup on prescribed drugs and
  on entities scispaCy extracts from the question.

### 3.3 Mode 3 prompt fix mattered more than expected

Notebook: `Mode_3_Prompt_Fix.ipynb`

The first Mode 3 prompt asked the LLM to "use the safety information below."
The model deferred to the doctor instead of using the KG facts. RAG+KG
accuracy was 35.2% on the safety-question subset — actually worse than RAG
alone, because the KG context made the LLM more cautious.

We rewrote the prompt to be diplomatic but directive: *"If the safety
information directly addresses the question, share it in plain language.
Recommending the patient confirm with their doctor is appropriate, but try to
be informative first rather than only deferring."*

Same KG content. Different prompt. Result: **35.2% → 82.6% on RAG+KG
accuracy on the safety-question subset (+47.4pp).** That single change is the
single biggest gain in the entire project.

### 3.4 Patient portal results

Notebooks: `Patient_Portal_Pipelines.ipynb`, `Patient_Portal_Router.ipynb`,
`Patient_Portal_TestSet_Eval.ipynb`.

Per-mode accuracy (each mode scored on every question):

| Mode | Train (n=250) | Test (n=100) |
|------|--------------:|-------------:|
| LLM only | 63.6% | 53.0% |
| LLM + RAG | 92.8% | 95.0% |
| LLM + RAG + KG | 95.2% | 97.0% |
| Oracle (best of 3) | 97.6% | 98.0% |

Routing strategies on the same data:

| Strategy | Train | Test |
|----------|------:|-----:|
| Always Mode 1 | 63.6% | 53.0% |
| Always Mode 2 | 92.8% | 95.0% |
| Always Mode 3 | 95.2% | 97.0% |
| **Router (MLP, 23 features)** | **94.0%** | **92.0%** |
| Oracle | 97.6% | 98.0% |

Same architecture as the failed MedQA router. Different result: the patient
portal router gets within 5pp of the best fixed mode and lets us skip
retrieval and KG calls on questions that don't need them.

Per-category accuracy (training set, the basis of Chart B in the poster):

- Mode 1 collapses on patient-specific categories: 13% on patient lifestyle,
  26% on patient indication.
- Mode 2 fixes patient questions: 100% on every patient category.
- Mode 3 is the only mode that lifts drug-interaction accuracy from 68% to
  85%. Drug interactions are the case where curated KG facts clearly help.

Router confusion matrix on the training fold (overall 93.2%):

| | Pred M1 | Pred M2 | Pred M3 |
|---|---:|---:|---:|
| Expected M1 | 73 (97%) | 0 | 2 (3%) |
| Expected M2 | 6 (6%) | 97 (92%) | 3 (3%) |
| Expected M3 | 5 (7%) | 1 (1%) | 63 (91%) |

The few costly errors are Mode-3 questions routed to Mode 1 (5 cases). Mode
2 ↔ Mode 3 mis-routes are cheap because both modes usually answer correctly.

### 3.5 Why it worked on patient portal but not on MedQA

Two reasons:

1. **Patient context is in the features.** The router sees the count of
   active medications, conditions, allergies, plus how many of the patient's
   prescribed drugs are mentioned in the question. These features
   meaningfully discriminate mode correctness on this task. On MedQA there is
   no patient — the features were question-only and weak.

2. **Mode coverage is better separated.** Mode 1 truly fails on
   patient-specific questions (13% on lifestyle); Mode 2/3 truly fails on
   nothing-in-chart questions. The "all 3 wrong" rate is 2% (vs 28% on
   MedQA). The "exactly one mode right" cases dominate the dataset, and the
   router can learn them.

---

## Part 4 — Dataset expansion (v2)

Notebook: `Patient_Portal_v2_Overnight.ipynb`. Generator:
`scripts/generate_v2_dataset.py`.

To stress-test the router at larger N, we expanded the dataset:

- 400 patients (50 original + 350 newly generated)
- 1600 training questions + 400 test questions (2000 total)
- Same 12 categories, matched to within ±0.1pp of original proportions
- 80/20 stratified split

Auto-labels (category-rule-based) cover all 2000. Real ground truth (running
each question through Modes 1/2/3 via Ollama and scoring) is being computed
overnight on Colab and will produce `mode_X_correct` flags + `winning_mode`
labels for the full 2000-question set. Estimated ~8 hours on a T4 GPU.

Once the overnight run completes, the v2 router will be retrained on the
1600 training questions and evaluated on the held-out 400.

---

## Part 5 — Deployment: Streamlit demo

Folder: `streamlit_app/`.

A live + mock-mode Streamlit UI that lets you pick any of the 50 synthetic
patients, see their chart, and ask a question. Two run buttons:

- **Run** — uses the MLP router to pick one mode and only calls the LLM
  for that mode (saves the other two LLM calls).
- **Run all 3 modes** — runs every mode and shows them side-by-side with
  the router's pick highlighted.

In live mode it loads MedCPT + scispaCy + PrimeKG + the trained MLP and
calls Gemma2 via Ollama. In mock mode it serves pre-computed responses from
the JSONL files for fast demoing.

---

## What we learned

Three reusable insights came out of this work, each tested twice (once on
MedQA, once on the patient portal):

1. **Cross-encoder gating is the most valuable single addition to any RAG
   pipeline.** Without it, retrieval can hurt more than it helps. With it,
   retrieval is the strongest single mode.

2. **Adaptive injection beats blanket injection.** On MedQA, KG injecting on
   9.5% of questions beat injecting on 87.5% by +3.5pp. On the patient
   portal, the same logic shows up in the router's behavior: it sends only
   questions that need KG to Mode 3.

3. **A learned router needs separable mode-correctness.** Where the modes
   overlap heavily (MedQA: 28% all-wrong + 28% all-correct), no router
   architecture we tried could find the signal. Where the modes have clear
   per-category specialization (patient portal), the same MLP architecture
   works well.

The negative MedQA result is part of the story, not in spite of it. It tells
us when a router will and won't work, and the patient-portal pivot tests that
prediction.

---

## Files

**Core notebooks (in `final-project-horus/`):**

- `RAG_MedQA_Gemma2_9b_MEDCPT.ipynb` — baseline
- `RAG_MedQA_Gemma2_9b_MEDCPT_upgraded.ipynb` — +6 fixes
- `PrimeKG_KG_SmokeTest.ipynb` — KG infrastructure
- `Cond5_v2.ipynb` — smart KG injection
- `MLP_Router_Training_v2.ipynb` — 23-feature router on MedQA
- `Patient_RAG_Builder.ipynb` — patient FAISS index
- `Patient_Portal_Pipelines.ipynb` — 3-mode pipeline
- `Mode_3_Prompt_Fix.ipynb` — the diplomatic prompt
- `Patient_Portal_Router.ipynb` — router on patient portal
- `Patient_Portal_TestSet_Eval.ipynb` — held-out eval

**Data (in `synthetic_patients/`):**

- `patients.jsonl` — 50 patients
- `questions.jsonl` + `test_questions.jsonl` — 250 + 100 labelled questions
- `patient_portal_responses_v2.jsonl` — Mode 1/2/3 answers
- `patient_portal_ground_truth_v2.jsonl` — `mode_X_correct` + `winning_mode`
- `patients_v2.jsonl` (400) + `questions_v2.jsonl` (1600) +
  `test_questions_v2.jsonl` (400) — expanded dataset

**Trained artefacts:**

- `patient_router/router_mlp.pt` + `scaler.pkl` — MLP weights
- `patient_index/patient_index.bin` + chunks — FAISS over patient records
- `primekg_index.pkl` — PrimeKG entity → triples lookup

**Deployment:**

- `streamlit_app/` — UI with live + mock modes

**Generator + analysis scripts (in `scripts/`):**

- `generate_v2_dataset.py` — expanded-dataset generator
- `make_poster_charts.py` — Chart A/B/C/D/E
- `make_flowchart.py` — architecture diagram
