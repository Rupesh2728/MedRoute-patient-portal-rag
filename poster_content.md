# Poster content — CSE 676A Final Project

Drop each section into the matching block of the UB template (48 × 36 in).
Word counts kept tight to fit the layout.

---

## TITLE

Hybrid RAG + Knowledge Graph Patient Portal with Adaptive MLP Routing

## SUBTITLE

A learned router that selects between LLM, retrieval, and knowledge-graph
augmentation for safe, patient-specific medical Q&A.

## AUTHORS

Maqsood [add co-authors]
CSE 676A — Deep Learning · Spring 2026 · University at Buffalo

---

## INTRODUCTION

Patient-portal questions ("Can I take ibuprofen with my prescriptions?")
require three different kinds of knowledge: (1) the patient's chart, (2) the
LLM's general medical knowledge, and (3) curated drug-safety facts. No single
augmentation strategy is best for every question — retrieval helps when the
chart matters, knowledge-graph (KG) injection helps for drug interactions, and
plain LLM answers suffice for general queries.

We build a 3-mode pipeline (LLM only / + patient RAG / + KG) and train an MLP
router to select the right mode per question. The system answers questions for
50 synthetic patients with realistic prescriptions, conditions, allergies, and
vitals — recovering most of the accuracy of the strongest mode at lower
compute cost.

---

## METHODS

**Data.** 50 hand-crafted synthetic patients; 250 training questions and 100
held-out test questions across 12 categories (drug info, mechanism, food/drug
interactions, side effects, indications, timing, monitoring, etc.).

**Three answer modes.**
- **Mode 1 — LLM only:** Gemma2-9B (Ollama) sees the patient's prescription
  list and the question.
- **Mode 2 — + Patient RAG:** MedCPT query encoder + FAISS over per-patient
  chunked records; cross-encoder reranks top-4 → top-3 chunks injected into
  the prompt.
- **Mode 3 — + Patient RAG + KG:** PrimeKG (≈4 M biomedical relations) lookup
  on prescribed drugs and on entities extracted from the question via
  scispaCy. A diplomatic-but-directive prompt encourages the LLM to use KG
  facts rather than defer.

**MLP Router.** 23 features → 32 → 16 → 3-way softmax.
Features: 12 question-type one-hots, 4 retrieval scores, 2 KG features,
3 patient features, 2 question features. Trained with stratified 5-fold CV.

---

## DATA ANALYSIS

**Chart A — Question-category distribution (250 train + 100 test).**
12 categories spanning drug-centric (general info, mechanism, interactions,
food, lifestyle, side effects) and patient-centric (indication, timing,
vitals, lifestyle, side effects, monitoring) Q&A.

**Chart B — Per-mode accuracy by category.**
Mode 1 fails on patient-specific categories (indication, vitals); Mode 2
recovers most patient questions; Mode 3 wins drug-interaction and food-safety
categories where curated facts matter.

**Chart C — Router confusion matrix (test set, n = 100).**
Diagonal-heavy. Most errors are mis-routes between Mode 2 and Mode 3 — both
of which usually answer correctly, so the end-to-end accuracy hit is small.

---

## RESULTS — CHART D

**Train (n = 250) vs Test (n = 100) accuracy, %**

| Strategy            | Train | Test |
|---------------------|------:|-----:|
| Always Mode 1       | 63.6  | 53.0 |
| Always Mode 2       | 92.8  | 95.0 |
| Always Mode 3       | 95.2  | 97.0 |
| **Router (MLP)**    | **94.0** | **92.0** |
| Oracle              | 97.6  | 98.0 |

(Render as grouped bars: train (light) vs test (dark), 5 groups.)

---

## RESULTS

The pipeline recovers most of the accuracy of the strongest single mode while
selectively skipping unneeded augmentation:

- **Mode 3** is the strongest single strategy after a prompt-fix that turned
  the LLM from "see your doctor" defaults to using the KG facts directly
  (35.2 % → 82.6 % on the RAG + KG eval set).
- **Router** achieves 94 % train / 92 % test — within 3–5 pp of always running
  Mode 3, while invoking the KG only on questions that need it.
- **Oracle** = 98 %, meaning fewer than 2 % of test questions stump all three
  modes — the headroom for a learned router is real but modest.

**Figure E — Compute vs accuracy.** Always-Mode-3 sits at the top of the
accuracy axis but the highest compute. The router trades ~3 pp accuracy for
roughly 1/3 fewer KG/retrieval calls.

---

## CONCLUSION

A 23-feature MLP router can learn to triage patient-portal questions among
LLM-only, RAG, and RAG + KG pipelines, recovering most of the accuracy of the
strongest mode at lower compute. Two design choices mattered most:

- **Diplomatic-but-directive Mode-3 prompt** turned the KG from inert context
  into actionable drug-safety information (+47 pp on RAG + KG accuracy).
- **Question-type features** (rather than raw keyword flags) made the router
  generalise: 92 % accuracy on a held-out 100-question test set with no
  retraining.

Limitations: synthetic patients only; English-only; KG is general (PrimeKG)
rather than payer-specific. Next steps: real-world MIMIC-III validation, a
PHI-anonymisation framework, and a richer router (transformer over question +
retrieved context).

---

## REFERENCES

1. Jin Q. et al. *MedCPT: Contrastive Pre-trained Transformers with Large-scale
   PubMed Search Logs for Zero-shot Biomedical Information Retrieval.*
   Bioinformatics, 2023.
2. Chandak P., Huang K., Zitnik M. *Building a knowledge graph to enable
   precision medicine (PrimeKG).* Scientific Data, 10, 67 (2023).
3. Neumann M. et al. *ScispaCy: Fast and Robust Models for Biomedical Natural
   Language Processing.* BioNLP Workshop, 2019.
4. Lewis P. et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP
   Tasks.* NeurIPS, 2020.
5. Google DeepMind. *Gemma 2: Improving Open Language Models at a Practical
   Size.* Technical Report, 2024.
6. Johnson J., Douze M., Jégou H. *Billion-scale similarity search with GPUs
   (FAISS).* IEEE Big Data, 2019.
