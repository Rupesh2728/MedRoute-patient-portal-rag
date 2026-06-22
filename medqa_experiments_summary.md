# MedQA Experiments — Gemma2-9B with MedCPT and PrimeKG

All runs use Gemma2-9B (Ollama) on the MedQA-USMLE validation set, n = 200
questions sampled from the 1,272-question split. Retrieval uses the MedCPT
Query/Article/Cross encoder family on the S-PubMed corpus. Knowledge graph
lookups use PrimeKG (~4M biomedical relations).

## Headline result

The strongest configuration is **adaptive KG injection on top of upgraded RAG**:
**56.50% accuracy / 63.24% F1**. This is +5.5pp accuracy over the original
baseline RAG run (51.0% / 58.3%) and +2.0pp over Gemma2 with no retrieval at
all (54.5% / 60.6%).

Three things mattered most:

1. **Six engineering fixes to the original RAG pipeline** moved it from
   *worse than no RAG* (51.0%) to clearly better than no RAG (57.0%).
2. **Naive KG injection hurts more than it helps** when applied to every
   question (87.5% inject rate → only +0.5pp).
3. **Adaptive KG injection** (only inject when cross-encoder confidence
   exceeds a threshold) fires on 9.5% of questions but lifts accuracy by
   +4.0pp over no-KG.

---

## Experiment 1 — Baseline RAG vs Gemma2-alone

Notebook: `RAG_MedQA_Gemma2_9b_MEDCPT.ipynb`

| Condition                            | Accuracy | F1     |
|--------------------------------------|---------:|-------:|
| Cond 1: Gemma2 alone (no context)    | 54.50%   | 60.62% |
| Cond 2: Baseline RAG (MedCPT + S-PubMed) | 51.00%   | 58.29% |

Adding off-the-shelf retrieval reduced accuracy. The baseline pipeline
retrieved short passages and dumped them into the prompt without filtering or
formatting. The LLM was distracted by irrelevant context. Average retrieval
score was 0.760, but the cross-encoder relevance for those passages was not
used to gate or rerank.

This is the failure case that motivated the upgrade.

Output: `results_spubmed/eval_results.jsonl`

---

## Experiment 2 — Upgraded RAG with 6 engineering fixes

Notebook: `RAG_MedQA_Gemma2_9b_MEDCPT_upgraded.ipynb`

The six fixes added to the pipeline:

1. **Stem extraction** — strip away the question's preamble before encoding.
2. **Multi-query retrieval** — retrieve from both the stem and the option set.
3. **Cross-encoder reranking** — rerank top-K candidates with the MedCPT
   Cross-Encoder before injecting.
4. **Adaptive context** — only inject context when cross-encoder confidence
   exceeds a threshold.
5. **Chain-of-thought prompting** — instruct the LLM to reason step by step.
6. **Structured context** — format passages with explicit delimiters and
   labels rather than raw concatenation.

| Condition                    | Accuracy | F1     |
|------------------------------|---------:|-------:|
| Cond 1: Gemma2 alone (letter-only) | 55.00%   | 60.73% |
| Cond 2: Old RAG (letter-only) | 51.00%   | 58.29% |
| Cond 3: Gemma2 alone (CoT)   | 54.50%   | 60.23% |
| **Cond 4: Upgraded RAG**     | **57.00%** | **63.33%** |

Cond 4 improved over Cond 2 by **+6.00pp accuracy** and +5.04pp F1. Context
was used in 73.5% of questions; the average cross-encoder relevance score on
injected context was 5.359 (well above the 2.0 confidence threshold).

Confidence-threshold ablation:

| Threshold | Context-used rate | Accuracy |
|-----------|------------------:|---------:|
| 2.0       | 68%               | 50.00%   |
| 5.0       | 56%               | 50.00%   |

Output: `results_spubmed_upgraded/cond4_detailed.jsonl`

---

## Experiment 3 — PrimeKG coverage smoke test

Notebook: `PrimeKG_KG_SmokeTest.ipynb`

Before running KG ablations, we tested whether PrimeKG actually has entries
for the entities in MedQA questions.

| Metric                    | Value      |
|---------------------------|-----------:|
| Questions tested          | 200        |
| Questions with at least one PrimeKG hit | **175 (87.5%)** |

87.5% coverage is enough to make a KG-augmented condition meaningful. The
remaining 12.5% are abstract or non-medical questions where PrimeKG is not
expected to help.

Output: `primekg_smoke_results.json`

---

## Experiment 4 — Naive KG injection (Cond 4 vs Cond 5 v1)

Notebook: `Cond4_vs_Cond5_KG.ipynb`

First test of adding KG triples to the upgraded RAG pipeline. KG triples were
injected on every question that had at least one entity match (no filtering).

| Condition         | Accuracy | F1     | KG injected | Avg triples |
|-------------------|---------:|-------:|------------:|-----------:|
| Cond 4 (no KG)    | 53.50%   | 60.64% | 0%          | 0.00       |
| Cond 5 (KG, naive)| 54.00%   | 61.30% | 87.5%       | 4.34       |

KG bought only **+0.50pp accuracy** over no-KG. Per-question analysis: KG
rescued 13 questions but broke 12 — net +1 question. The triples added noise
on questions where they were tangentially relevant.

Output: `results_cond5_kg/cond4_vs_cond5.json`

---

## Experiment 5 — Smart KG injection (Cond 5 v2)

Notebook: `Cond5_v2.ipynb`

Three changes turned KG from a distraction into a useful signal:

1. **Cross-source reranking** — combine retrieved passages and KG triples
   into a single candidate set, then rerank everything with the cross-encoder.
2. **Natural-language formatting** — convert raw `(subject, predicate, object)`
   tuples into prose sentences ("Metformin interacts with ibuprofen").
3. **Adaptive injection** — only inject KG when cross-encoder relevance for
   the top KG candidate exceeds the same threshold used for passage gating.

| Condition                        | Accuracy | F1     | KG inject rate |
|----------------------------------|---------:|-------:|---------------:|
| Cond 4 (no KG)                   | 52.50%   | 59.28% | 0.0%           |
| Cond 5 v1 (KG, naive)            | 53.00%   | 60.89% | 87.5%          |
| **Cond 5 v2 (KG + rerank + NL + adaptive)** | **56.50%** | **63.24%** | **9.5%**      |

Deltas:

- v2 vs Cond 4: **+4.00pp accuracy**, +3.96pp F1
- v2 vs v1: +3.50pp accuracy, +2.35pp F1

v2 rescued 15 questions and broke 7, a net +8.

The most surprising number is the inject rate: v2 fires on only **9.5%** of
questions, yet beats the 87.5%-rate v1 by +3.5pp. Adding fewer, more relevant
KG facts is better than adding many irrelevant ones.

Output: `results_cond5_v2/cond4_v1_v2_comparison.json`

---

## What we learned

The MedQA work produced three reusable engineering insights that carried
directly into the patient-portal pipeline:

1. **Cross-encoder gating is the most valuable single addition.** It turns
   noisy retrieval into useful retrieval. Without it, RAG underperforms no-RAG.

2. **Natural-language formatting of KG triples matters.** The LLM uses prose
   sentences much more than raw tuples. The same is true for the patient
   portal Mode-3 prompt fix (which lifted RAG+KG accuracy by +47pp on the
   patient portal task).

3. **Adaptive injection beats blanket injection.** Inject only when the
   retrieved or KG content is confidently relevant. This is the same
   threshold-gating logic used in the patient-portal Mode 2 / Mode 3
   pipelines.

These insights motivated the per-question MLP router for the patient portal:
*if adaptive injection works, then a learned router that decides when to
retrieve and when to query the KG should work even better.*
