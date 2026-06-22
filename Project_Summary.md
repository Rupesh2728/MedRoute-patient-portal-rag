# Hybrid RAG for Medical QA — Project Summary

**CSE 676A Final Project · Spring 2026**

Team: Ishaq Miyawala (ishaqibr), Maqsood Ahmed (m58), Rupesh Chowdary (rupeshch)

---

## Phase 1: Baseline Diagnosis

The baseline notebook (`RAG_MedQA_Gemma2_9b_MEDCPT.ipynb`) showed RAG (51%) **underperforming** Gemma2-9B alone (54.5%) on MedQA. We traced this to five root causes in the code:

1. **MedCPT query encoder truncated at 64 tokens** — USMLE vignettes are 200+ tokens, so the actual question was never seen by the encoder
2. **Only question text used for retrieval** — drug/disease names in MC options were ignored despite being the most retrievable terms
3. **No reranking after FAISS** — top-7 passages went straight to the prompt, including noise
4. **Letter-only prompt with `max_tokens=16`** — the model had no room to reason
5. **Context always injected** — even when retrieval scores were low, noisy passages were fed in

## Phase 2: Upgraded Pipeline

**Notebook:** `RAG_MedQA_Gemma2_9b_MEDCPT_upgraded.ipynb`

Six targeted fixes:
- Stem extraction (last 1–2 sentences fit in 64 tokens)
- Multi-query retrieval (stem + per-option entities)
- MedCPT cross-encoder reranking (512-token window)
- Chain-of-thought prompting with `max_tokens=512`
- Adaptive context injection (skip if cross-encoder score < 2.0)
- Structured context blocks with KG/textbook separation

**Result: 57.0% accuracy / 63.3% F1** — a +6pp improvement over baseline RAG and +2pp over Gemma2-alone. The fix worked; RAG no longer hurts.

## Phase 3: Wikidata KG (Failed)

We attempted to integrate the existing `kg_retriever.py` (Wikidata SPARQL). Two problems surfaced:
- **Rate limiting** (HTTP 429) on every call
- **Broken entity extraction** — regex word-splitting produced `'59yearold'`, `'with'`, `'lysosomal'` (fragments and stop words)

scispaCy with the BC5CDR NER model fixed entity quality but Wikidata's medical coverage remained insufficient (~14% of questions returned ≥1 useful triple).

## Phase 4: PrimeKG Integration

Switched to PrimeKG (Harvard, 2023) — a purpose-built medical knowledge graph with 4M relations from 17 sources (DrugBank, MONDO, HPO, DisGeNET).

**Notebooks:** `PrimeKG_KG_SmokeTest.ipynb`, `Cond4_vs_Cond5_KG.ipynb`

- Coverage jumped from 14% (Wikidata) to **87.5%** (PrimeKG)
- 17,080 diseases, 7,957 drugs, 28,000 genes/proteins
- Single CSV download, no API rate limits

But end-to-end accuracy gain was marginal: Cond5 (with KG) only +0.5pp over Cond4 (without KG). 13 rescues vs 12 breaks — essentially noise.

## Phase 5: KG Improvements (Cond 5 v2)

**Notebook:** `Cond5_v2.ipynb`

Three targeted fixes:
1. **Cross-encoder reranking for triples** — same MedCPT cross-encoder applied to (question, triple) pairs
2. **Adaptive KG injection** — skip the KG block if top reranked triple's score is below threshold
3. **Natural-language formatting** — `Metformin is indicated for diabetes` instead of `metformin -> indication -> diabetes`

**Result:** Cond 5 v2 hit 56.5% accuracy / 63.2% F1, showing +4pp over Cond 4. KG injected for only 9.5% of questions — the adaptive switch correctly suppressed the KG block when triples weren't relevant.

## Phase 6: MLP Router

**Notebooks:** `MLP_Router_Training.ipynb` (12 features), `MLP_Router_Training_v2.ipynb` (23 features)

**Architecture:**
- Input: 23 features per question
  - 12 retrieval features (cross-encoder scores, candidate counts)
  - 6 question-type binary flags (treatment, diagnosis, mechanism, side effect, lab interpretation, anatomy)
  - 5 entity features (chemical/disease counts, options-are-drugs, top KG relation type, mean option length)
- Hidden layers: `23 → 48 → 24 → 3` with ReLU + Dropout(0.25)
- Output: 3-class softmax over modes [Parametric, Textbook, Hybrid]
- Loss: cross-entropy with soft labels (split mass when multiple modes correct)
- Training: 5-fold stratified CV, Adam, ~960 parameters

**Three modes:**
- **P (Parametric):** Gemma2 + CoT, no context
- **T (Textbook):** RAG with adaptive textbook injection
- **H (Hybrid):** RAG + PrimeKG triples with cross-encoder reranking

**Held-out evaluation results:**
| Configuration | Accuracy |
|---|---|
| Always P (parametric) | 54.5% |
| Always T (textbook) | 54.5% |
| Always H (hybrid) | 55.0% |
| MLP Router + self-consistency | 55.5% |
| Oracle (any mode correct) | 65.0% |

**The router collapsed to "always P" for 96.5% of held-out questions.**

## Honest Finding

On Gemma2-9B, the routing decision degenerates because the modes are too similar. From the training data:
- 40.6% of questions: all 3 modes correct (no preference to learn)
- 31.4% of questions: all 3 modes wrong (no signal at all)
- 28% of questions: mixed correctness (the only routable subset)

**The MLP correctly learned that parametric mode is usually best.** This is consistent with the diminishing returns hypothesis from prior work (Adaptive-RAG, HybridRAG, Medical Graph RAG) — adaptive routing is most valuable for smaller models with weaker parametric coverage.

## What Worked

- **Upgraded retrieval pipeline:** +6pp over baseline (51% → 57%)
- **PrimeKG infrastructure:** 87.5% coverage, single download, no rate limits
- **scispaCy NER:** 4–5× better entity quality than regex
- **Cross-encoder reranking:** for both passages AND triples
- **Adaptive context injection:** prevents noisy retrieval from poisoning the prompt

## What Didn't Work

- **Wikidata KG:** rate-limited and sparse for medical concepts
- **KG contribution on Gemma2-9B:** marginal (+0–1pp); the model already knows
- **MLP router on Gemma2-9B:** collapses to parametric; no routable signal at this scale

## Notebooks Produced

| Notebook | Purpose |
|---|---|
| `RAG_MedQA_Gemma2_9b_MEDCPT_upgraded.ipynb` | 6-fix upgraded pipeline |
| `PrimeKG_KG_SmokeTest.ipynb` | KG infrastructure + coverage test |
| `Cond4_vs_Cond5_KG.ipynb` | Naive KG injection comparison |
| `Cond5_v2.ipynb` | Improved KG (rerank, NL, adaptive) |
| `MLP_Router_Training.ipynb` | Router v1 (12 features) |
| `MLP_Router_Training_v2.ipynb` | Router v2 (23 features) |
| `MLP_Router_Training_v2_2B.ipynb` | Same v2 pipeline on Gemma2-2B |
| `Cross_Domain_Router_Demo.ipynb` | Mixed MedQA + MMLU benchmark |

## Project Story (defensible writeup framing)

We constructed an end-to-end hybrid RAG + KG pipeline for medical question answering on MedQA, evaluated on Gemma2-9B. Our contributions:

1. **Diagnosed and fixed** five root causes of baseline RAG failure (+6pp over baseline)
2. **Replaced the unreliable Wikidata KG** with PrimeKG, achieving 87.5% coverage on USMLE
3. **Introduced cross-source mutual ranking** — using the MedCPT cross-encoder to score (question, triple) pairs, an under-explored design point in prior work
4. **Trained an MLP router** with 23 features for adaptive mode selection (parametric / textbook / hybrid)
5. **Reported the honest finding** that on Gemma2-9B, the router converges to always-parametric, validating the diminishing-returns hypothesis for adaptive retrieval at scale

## Future Work

1. **LoRA fine-tune Gemma2-2B** on MedQA train (+5–10pp expected)
2. **Self-consistency stack** — N=5 samples with majority vote (+3–5pp)
3. **Smaller-model demonstration** — Gemma2-2B where router has actual signal
4. **Few-shot CoT** — 3 worked USMLE examples in every prompt (+2–4pp)
5. **End-to-end fine-tuning of cross-encoder** for MedQA passage relevance

## Key Numbers

| Metric | Value |
|---|---|
| Baseline RAG accuracy | 51.0% |
| Gemma2-9B alone | 54.5% |
| **Upgraded pipeline** | **57.0%** |
| Cond 5 v2 (KG + improvements) | 56.5% |
| Router (held-out) | 55.5% |
| Oracle ceiling | 65.0% |
| PrimeKG coverage | 87.5% |
| Trainable subset of training data | 28% |
| Router mode distribution (held-out) | 96.5% P, 3.5% T, 0% H |
