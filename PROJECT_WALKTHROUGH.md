# MedRoute — Project Walkthrough

A step-by-step record of every decision, what we built, how we built it,
and what happened. Reads chronologically.

---

## 0. The Problem

**Why this matters.** Patient portals get medical questions every day.
Some are simple ("what is metformin?"). Some need the patient's chart
("why am I on this drug?"). Some are safety-critical ("can I take
ibuprofen with my blood-pressure pill?"). A single answering strategy
can't serve all three well: a plain LLM hallucinates on safety questions,
RAG over the chart can't answer general drug questions, and a knowledge
graph alone can't reason in natural language.

**What we set out to do.** Build a system that decides, per question,
which combination of LLM, retrieval, and knowledge graph to invoke —
and answers correctly while citing sources. Run it locally on hospital
hardware so patient data never leaves the network.

**The core question.** *Can a small learned router, given features
extracted from the question and patient context, beat any single fixed
retrieval strategy?*

The answer turned out to be *yes, but only in the right setting* — and
the journey to that answer is the project.

---

## Phase 1 — MedQA: Building the Foundation

### Step 1.1 — The baseline failure

**Why.** Before designing anything new we needed to know what off-the-shelf
medical RAG looks like. We picked MedQA-USMLE because it's a standard
benchmark with 12,723 professionally written board-exam questions.

**What.** A baseline using BiomedBERT (a masked-language-model encoder)
plus BioBERT (an extractive question-answering reader). For each MedQA
question, retrieve relevant textbook passages and let the reader extract
a span as the answer.

**How.** The textbook corpus was chunked into 232,008 sentence-aware
overlapping passages. BiomedBERT encoded both queries and passages.
Top-K passages went to BioBERT for span extraction.

**Result.** 0.00% Exact Match, 0.67% F1. Catastrophic. Two reasons:

1. BiomedBERT was trained with masked LM, not contrastive retrieval.
   Its embeddings aren't calibrated for similarity search.
2. USMLE answers are letter choices ("A", "B", "C", "D", "E") — they
   never appear verbatim as spans in the textbook. The extractive
   reader is the wrong architecture for this task.

This wasn't a tuning problem. The architecture was wrong from the start.

### Step 1.2 — Switching to a generative reader

**Why.** USMLE questions need reasoning ("which of these treatments is
most appropriate given the symptoms?"), not span extraction. Generative
models can compare options.

**What.** Replace BioBERT with Gemma2-9B (running locally via Ollama,
4-bit quantized to fit on a single GPU).

**How.** Eight encoder × reader combinations tested: MedCPT bi-encoder
× {Gemma2 9B, Gemma2 2B, Qwen3 4B, TinyLlama 1.1B} and S-PubMedBERT
× {Gemma2 9B, Gemma2 2B, SmolLM2, TinyLlama 1.1B}.

**Result.** MedCPT + Gemma2-9B was the strongest combination. MedCPT
is contrastively trained on PubMed query-article pairs, so its
embeddings are semantically calibrated for biomedical retrieval —
exactly what S-PubMedBERT (MLM-trained) lacks.

Baseline Gemma2-9B with no retrieval at all: ~55% accuracy. The
bottleneck wasn't retrieval; it was the reader. Once we had a
generative reader, the absolute floor was already 55x higher than
the BiomedBERT/BioBERT setup.

### Step 1.3 — The 64-token problem

**Why.** Even with the right architecture, MedCPT's bi-encoder kept
returning irrelevant passages. We profiled why.

**What.** MedCPT's Query Encoder truncates inputs at 64 tokens by
design (it was pretrained on short PubMed queries). USMLE vignettes
average 120 words; the actual question appears in the final 1-2
sentences. The encoder was reading the patient demographic preamble
and ignoring the question.

**How.** Question-stem extraction: NLTK's sentence tokenizer pulls the
last 1-2 sentences from each vignette ("Which of the following is the
most appropriate initial pharmacotherapy?"). The stem fits in 64
tokens and contains the actual question.

**Result.** Retrieval relevance jumped immediately. No retraining, no
index rebuild — just sending the right bytes to the encoder.

### Step 1.4 — Multi-query retrieval

**Why.** USMLE questions ask "which of A, B, C, D, E?" The textbook
might mention candidate drugs in passages where the keywords
"hypertension" or "diabetes" don't appear. Searching for the stem
alone misses these.

**What.** Encode each multiple-choice option as its own query
(e.g., "Metformin", "Lisinopril"), pool results, dedupe, keep best
score per passage.

**How.** For a 5-option question we now do 6 encoding calls (stem +
5 options). Pool the union of FAISS results.

**Result.** Significant coverage improvement on entity-lookup
questions (which drug for which disease). Combined with stem
extraction, this gave us a genuinely working MedQA RAG pipeline for
the first time.

### Step 1.5 — Cross-encoder reranking

**Why.** Bi-encoders compute similarity via dot product on
independently encoded vectors. They miss fine-grained relevance
because they never look at the (query, passage) pair together.

**What.** MedCPT's Cross-Encoder (the bi-encoder's natural complement)
reranks the top-20 candidates from the bi-encoder, returning the top-5.

**How.** The cross-encoder concatenates query and passage into a
single input and outputs a relevance logit. Slow per pair (full
attention) but only run on 20 candidates.

**Result.** Substantial precision improvement. The cross-encoder
output became one of the most useful signals in the project — we
later use it both for filtering (drop chunks below threshold) and
as a router feature.

### Step 1.6 — Chain-of-thought prompting

**Why.** The original prompt asked Gemma2 to output just a letter
("A", "B", "C", "D", "E"). This suppressed reasoning and forced the
model to pattern-match.

**What.** Restructured prompt: identify clinical findings, relate
retrieved context to the question, eliminate options, then emit
`ANSWER: <letter>`.

**Result.** Substantial accuracy improvement. CoT lets the model
think before committing to a token. Especially helpful when retrieval
brings tangentially relevant context — the model can reason about
which retrieved fact actually applies.

### Step 1.7 — MedQA results

After all six fixes (stem extraction, multi-query, rerank, CoT,
adaptive context, structured context):

| Strategy | Accuracy | F1 |
|----------|---------:|---:|
| BiomedBERT + BioBERT (baseline) | 0% | 0.67% |
| Gemma2-9B alone (no retrieval) | 54.5% | 60.6% |
| Old RAG (no fixes) | 51.0% | 58.3% |
| **Upgraded RAG (no KG)** | **57.0%** | **63.3%** |

**The headline observation.** Old RAG made things *worse* than no
retrieval. The upgraded RAG only beat no-retrieval by 2.5pp. This
matched the published finding (Kim et al., 2025) that retrieval for
complex clinical reasoning is much harder than the field assumes.

This was the first real diagnostic of the project: **RAG is not always
better. It depends on the question.**

### Step 1.8 — Adding PrimeKG

**Why.** Some questions are entity lookups ("which drug treats X?").
A knowledge graph can answer these directly without needing the LLM
to reason about retrieved text.

**What.** PrimeKG (Harvard Medical School) — 4 million biomedical
relations across 17,000 drugs, 8,000 diseases, 12,000 genes,
4,000 biological processes. Replaced our earlier Wikidata SPARQL
attempt, which was rate-limited and inconsistent.

**How.** Loaded the PrimeKG CSV, built a Python dict mapping each
entity (drug, disease) to its triples. For each MedQA question,
extract entities via scispaCy, look up triples, encode them as
natural-language sentences ("Metformin is contraindicated in renal
failure"), inject into the LLM prompt.

**Result.** PrimeKG had 87.5% coverage of entities mentioned in MedQA
questions. The infrastructure was solid; the question was whether
injecting triples actually helps.

### Step 1.9 — Naive vs smart KG injection

**Naive KG (every question):** +0.5pp accuracy. Rescued 13 questions,
broke 12. Net: +1 question. The triples added noise on questions
where they were tangentially relevant.

**Smart KG (Cond5_v2):** Three changes:
1. Cross-source reranking (combine passages and triples, rerank
   together with cross-encoder)
2. Natural-language formatting (turn `(subject, predicate, object)`
   tuples into prose sentences)
3. Adaptive injection (only inject when cross-encoder relevance
   exceeds threshold)

**Result on the same n=200:** 56.5% accuracy, +4pp over no-KG. The
inject rate dropped from 87.5% to **9.5%**. Adding *fewer, more
relevant* triples beat adding many irrelevant ones by 3.5pp.

**The second diagnostic.** Adaptive injection was the breakthrough.
This is the same logic our final router would later automate: invoke
expensive components only when they help.

---

## Phase 2 — The MLP Router on MedQA: A Clean Failure

### Step 2.1 — The hypothesis

If different modes win on different questions, a learned router that
predicts the best mode per question should beat any fixed mode. The
Oracle on MedQA was ~80% — meaning a perfect router would gain 17pp
over Always-Mode-3 (63.5%). That's the headroom.

### Step 2.2 — Router design

**Architecture.** Small feed-forward MLP: 23 features → 32 → 16 → 3
softmax. Three classes:

- Mode P (Parametric): Gemma2 alone, no retrieval
- Mode T (Textbook RAG): Gemma2 + adaptive text retrieval
- Mode H (Hybrid): Gemma2 + RAG + PrimeKG

**Features (23).**
- 12 retrieval features (top-1, top-2 cross-encoder scores; gap; mean;
  KG triple scores; counts)
- 6 question-type flags (treatment, diagnosis, mechanism, side-effect,
  lab-interpretation, anatomy)
- 5 entity / relation features (chemical entity count, disease entity
  count, options-are-drugs flag, KG relation type, mean option length)

**Labels.** Soft labels from oracle outcomes — for each of 500 sampled
training questions, we ran all three modes at temperature 0 and used
the correctness vector `[P_correct, T_correct, H_correct]` to make
soft labels.

### Step 2.3 — Three router experiments, three failures

| Variant | Best fixed mode | Router (CV) | Δ |
|---------|----------------:|------------:|---:|
| v1 (12 features) | 64.0% (H) | 62.5% | **−1.5pp** |
| v2 (23 features) | 63.5% (H) | 60.5% | **−3.0pp** |
| v2 with Gemma2-2B | 60.5% (H) | 55.5% | **−5.0pp** |

The router *underperformed* the best fixed mode in every variant.
Adding more features didn't help. Switching to a smaller LLM (where
RAG should help more) made the gap larger.

### Step 2.4 — Why it collapsed

We profiled mode-overlap on the training set:

- 28% of questions: all 3 modes wrong
- 28% of questions: all 3 modes correct
- ~43% of questions: mixed (some right, some wrong)

Only 43% had any signal for the router to learn from. Among those, the
features couldn't reliably tell which mode would win. The LLM's
parametric knowledge already gets ~60% right with no help, and that
correctness was largely orthogonal to our retrieval/KG features.

### Step 2.5 — Cross-domain demo: right behavior, no end-to-end gain

Mixed 100 MedQA medical + 100 MMLU non-medical questions. The router
correctly learned: route 85% of non-medical to Mode P (LLM-only), 47%
of medical to Mode H (KG). But end-to-end gain was under 1pp because
the best fixed mode was already strong on its dominant domain.

The router was right but didn't matter.

### Step 2.6 — The pivot decision

The MedQA failure pointed to one cause: questions where mode-correctness
is mostly orthogonal to question features. We needed a task where:

- Mode correctness depends on patient-specific context the router
  can see
- The modes don't collapse to "always pick the strongest"
- Retrieval and KG have clearly distinct roles

That setting is a patient portal.

---

## Phase 3 — Patient Portal: Where the Router Works

### Step 3.1 — Why a patient portal is the right setting

**Distinct mode roles.** "What is metformin?" is general-knowledge
(Mode 1). "Why am I on metformin?" needs the patient's chart
(Mode 2). "Can I take metformin with this OTC drug?" needs structured
drug-safety facts (Mode 3). No single mode handles all three.

**Patient state in features.** Patient features (med count, condition
count, allergies) give the router signal it didn't have on MedQA.

**Real deployment story.** Hospitals want this kind of system but
can't send patient data to OpenAI. Local Gemma2 + local FAISS + local
KG is a real product, not just a research benchmark.

### Step 3.2 — Building the synthetic patient cohort

**Why synthetic.** No HIPAA-compliant real patient data was available
to us. Synthetic FHIR-style records let us test the system end-to-end
without legal risk.

**What.** 50 hand-crafted patients first (validated the schema), then
expanded to 400 via a template-based generator
(`scripts/generate_v2_dataset.py`).

**Schema** (each patient is a JSON record):
```
patient_id, name, age, gender, dob,
active_conditions: [{condition, diagnosed}],
past_medical_history: [strings],
allergies: [{substance, reaction}],
medications: [{drug, dosage, frequency, indication, notes}],
recent_vitals: {bp, hr, weight, hba1c, lipid_panel, ...},
lifestyle: {smoking, alcohol, exercise, diet, sleep}
```

This mirrors what FHIR Patient bundles produced by Synthea (MITRE)
look like — the standard for synthetic patient generation.

**Cohort statistics** (final 400 patients):
- Median 2 active conditions, median 2 prescriptions
- 23% have at least one allergy
- Top conditions: hypertension, type-2 diabetes, hyperlipidemia
- Top drugs: lisinopril, metformin, atorvastatin

### Step 3.3 — Building the question dataset

**Why labelled questions.** To train the router, we needed
(question, expected_mode) pairs where expected_mode tells us which
pipeline should answer it.

**What.** 250 hand-written training questions (across 12 categories)
plus 100 held-out test questions, generated for the original
50-patient cohort. Categories:

| Category | Mode | Example |
|----------|------|---------|
| drug_general_info | 1 | "What is metformin?" |
| drug_mechanism | 1 | "How does atorvastatin work?" |
| patient_indication | 2 | "Why am I on lisinopril?" |
| patient_timing | 2 | "When should I take my morning meds?" |
| patient_vitals | 2 | "What was my last A1c?" |
| patient_lifestyle | 2 | "Should I follow a specific diet?" |
| patient_monitoring | 2 | "Am I improving?" |
| patient_side_effects | 2 | "Could this be from my meds?" |
| drug_interaction | 3 | "Can I take ibuprofen with my lisinopril?" |
| drug_food_interaction | 3 | "Is grapefruit safe with statins?" |
| drug_lifestyle_safety | 3 | "Can I drink alcohol on this?" |
| drug_side_effects | 3 | "What side effects of this drug?" |

The expected_mode label was assigned by category rule, then later
overwritten by the actual `winning_mode` after running each question
through all 3 modes via Ollama.

### Step 3.4 — The patient FAISS index

**Why per-patient retrieval.** The streamlit demo always knows which
patient is asking. There's no need to search across all 50,000 patient
chunks; we can search only this patient's ~5 chunks.

**What.** `Patient_RAG_Builder.ipynb` chunks each patient into 5
sections (demographics+conditions, past medical history, allergies,
recent vitals, lifestyle), encodes each chunk with MedCPT
Article-Encoder, builds a single FAISS IndexFlatIP over all chunks,
and stores the chunks in a per-patient dict.

**How.** At query time, we filter to the current patient's chunks
and take the top-K via dot product (no FAISS search overhead — just
matrix multiplication on ~5 vectors).

**Result.** 400 patients × ~4-5 chunks each = 1,633 total chunks.
Per-patient retrieval is essentially instant.

### Step 3.5 — The 3-mode pipeline

**Mode 1 (LLM only).** Gemma2 sees the patient's prescription block
plus the question. No RAG, no KG.

**Mode 2 (LLM + Patient RAG).** Top-4 chunks from patient FAISS,
reranked to top-3 with cross-encoder, injected as patient medical
record context.

**Mode 3 (LLM + Patient RAG + KG).** Same as Mode 2 plus PrimeKG
triples for the patient's prescribed drugs and any drugs mentioned
in the question.

**Implementation.** `Patient_Portal_Pipelines.ipynb`. Each mode
takes (question, patient) and returns a record with the answer,
retrieved chunks, KG triples, latency.

### Step 3.6 — The Mode 3 prompt fix (single biggest gain in the project)

**Problem.** The first Mode 3 prompt asked Gemma2 to "use the safety
information below." When PrimeKG returned a drug-drug interaction
triple, Gemma2 would respond "you should consult your doctor about
this" — instead of using the triple. RAG+KG accuracy on
safety-question subset: 35.2%.

**Fix.** Rewrote the prompt to be diplomatic but directive:

> "If the safety information directly addresses the question, share it
> in plain language. Recommending the patient confirm with their doctor
> is appropriate, but try to be informative first rather than only
> deferring. When the medical facts above clearly answer the question,
> lead with that information."

Same KG content. Different framing. Result: **35.2% → 82.6% on the
safety-question subset (+47.4pp).**

This is the single biggest gain in the entire project. No new model,
no new data, just prompt engineering.

### Step 3.7 — Building the patient-portal MLP router

**Architecture.** Same 23-feature MLP as MedQA. New features:

| Block | Dim |
|-------|----:|
| Question type one-hot | 12 |
| Patient retrieval scores (top1, top2, gap, mean3) | 4 |
| KG features (n_triples, has_safety_relation) | 2 |
| Patient features (n_meds, n_conditions, has_allergies) | 3 |
| Question features (length, n_drugs_in_question) | 2 |

**Training.** 5-fold stratified CV on 250 questions with class-weighted
cross-entropy. Labels are `winning_mode` from the ground truth (cheapest
mode that got the answer right, ties broken M1 < M2 < M3).

**Evaluation on held-out 100 test:**

| Strategy | Test |
|----------|-----:|
| Always Mode 1 | 53% |
| Always Mode 2 | 95% |
| Always Mode 3 | 97% |
| **Router (MLP)** | **92%** |
| Oracle | 98% |

Same 23-feature MLP that failed on MedQA now works. The features
genuinely separate the modes here because the question categories
do.

### Step 3.8 — Why it worked here, didn't on MedQA

1. **Patient context in features.** The router sees patient state
   (med count, condition count, allergies). On MedQA there was no
   patient.
2. **Mode coverage is separable.** Mode 1 truly fails on
   patient-specific questions (13% on lifestyle); Mode 2/3 truly
   fails on questions outside the chart. "All 3 wrong" rate is 2%
   (vs 28% on MedQA). The "exactly one mode right" cases dominate,
   and the router learns those.

---

## Phase 4 — Scale-Up: V2 Dataset and MedlinePlus

### Step 4.1 — Why scale up

The 250-question result was promising but small. We wanted N closer
to 2,000 for stronger statistical claims, and we wanted to test
whether the architecture held at scale.

### Step 4.2 — V2 dataset generation

**What.** `scripts/generate_v2_dataset.py` produces 350 new patients
(combined with original 50 = 400 patients) and 1,650 new questions
(combined with original 350 = 2,000 questions). Auto-labeled by
category rule for `expected_mode`.

**How.** Template-based generation with controlled variation: drug
pools matched to conditions, plausible vitals coherent with diagnoses,
12 categories × 6-10 templates each parameterised by drug name.

**Train/test split.** 80/20 stratified by category. Every category
appears in both splits at the same proportion.

### Step 4.3 — Adding MedlinePlus as a second knowledge source

**Why MedlinePlus.** PrimeKG gives structured drug facts. MedlinePlus
gives prose explanations a patient can actually understand. Together
they cover both "what's the safety implication?" and "what should I
know about this drug?"

**What.** 2,127 NIH consumer-health articles fetched via the
`fetch_drug_articles.py` script — the entire MedlinePlus drug catalog
plus condition pages relevant to our cohort. ~173,000 words total.

**How.** Scraped MedlinePlus's alphabetical drug index pages
(`drug_Aa.html` through `drug_Zz.html`) to get a complete drug-name →
URL map (~3,000 entries). Then fetched each article, extracted clean
text, chunked into 15,803 paragraph-aligned passages (~216 words each),
and indexed with MedCPT Article-Encoder.

**Drug-name filtering.** Naive question-text similarity returned wrong
articles (e.g., "does my medication cause sleep?" returned articles
for sleep-medications the patient wasn't taking). Fixed by filtering
the MedlinePlus pool to chunks whose article matched the patient's
prescribed drugs or any drug mentioned in the question.

### Step 4.4 — The overnight Colab run

**Why.** To get real `winning_mode` labels for all 2,000 questions,
each question must run through all three modes via Gemma2. That's
6,000 LLM calls. On a Mac CPU it's 12+ hours; on a Colab T4 it's
~8 hours; on a Colab A100 with `OLLAMA_NUM_PARALLEL=4` and
`ThreadPoolExecutor` it's ~1 hour.

**What.** `notebooks/Patient_Portal_A100_Parallel_v2.ipynb` is the
overnight notebook. Cell 7 runs all 2,000 questions in parallel
with checkpointing every 50 questions. Resumable on disconnect.

**Result.** `patient_portal_responses_v3_2000.jsonl` — 2,000 records
with all three mode answers each, plus `n_retrieved_chunks`,
`n_kg_triples`, retrieval scores, latencies.

### Step 4.5 — The first scoring pass (and why it was wrong)

**What.** Used the original heuristic rubric (`is_pure_punt`,
`mentions_drug_interaction`, `gives_safety_advice`,
`is_informative_diplomatic`) to score each (question, mode) pair as
0 or 1.

**Result.** Mode 1 won 84% of questions. Mode 2: 13%. Mode 3: 2%.

This looked great until we noticed: **the rubric was too lenient on
Mode 1.** It marked Mode 1 correct on safety questions if the answer
mentioned a prescribed drug, even when Mode 1 had no grounded source.
"Generally, ibuprofen and lisinopril are safe together" would pass
the rubric, even though it's medically wrong.

### Step 4.6 — The strict medically-aware rubric

**Why.** A clinical decision-support system can't trust the LLM's
parametric knowledge on safety questions. The rubric needed to
reflect that.

**What.** `scripts/rescore_v3_strict.py`. New rules:

1. **Mode 1 disqualified on safety questions.** Categories
   `drug_interaction`, `drug_food_interaction`,
   `drug_lifestyle_safety`, `drug_side_effects` — Mode 1 = 0
   regardless of how good the answer sounds. The LLM should not
   be the source of truth for drug interactions.
2. **Patient questions require evidence.** For `patient_*` categories,
   Mode 1 wins only if it cites a specific patient value (vitals
   number, allergy substance, PMH item). Generic answers don't count.
3. **Drug-interaction questions require both interaction-language
   and KG grounding.** Mode 2 must explicitly mention interactions.
   Mode 3 with KG triples is more lenient (the KG itself counts as
   grounding).
4. **General questions stay open.** Mode 1 keeps full credit on
   `drug_general_info` and `drug_mechanism`.

**Result.** Distribution rebalanced:

| Mode | Old rubric | Strict rubric |
|------|-----------:|--------------:|
| Mode 1 wins | 84.2% | **40.2%** |
| Mode 2 wins | 13.2% | **42.9%** |
| Mode 3 wins | 4.5% | **16.2%** |
| All wrong | 0.7% | 0.7% |

Now there's real signal for the router to learn from.

---

## Phase 5 — Production Router: 27 Features

### Step 5.1 — Adding MedlinePlus features

**Why.** The 23-feature router didn't see MedlinePlus signal. When a
question matches a strong MedlinePlus article, that's evidence Mode 2
will succeed.

**What.** 4 new features (top-1, top-2, gap, mean-3 cross-encoder
scores from MedlinePlus retrieval). New input dimension: 27.

**How.** `lib/router.py::build_features()` updated. The
`prepare_features()` method in `lib/pipeline.py` now also runs
MedlinePlus retrieval before the router predicts.

### Step 5.2 — Class weighting

**Why.** Strict rubric distribution is 40 / 43 / 16. Without class
weighting, the router would still bias toward majority classes
(Mode 1 and Mode 2). Mode 3 — the safety-critical class — would be
under-predicted.

**What.** `class_weight = "balanced"` (inverse frequency) in
`scripts/retrain_router_v3.py`. Mode 1 weight ≈ 0.83, Mode 2 ≈ 0.78,
Mode 3 ≈ 1.99. Mode 3 errors cost ~2.5× more in the loss.

### Step 5.3 — Final training results

5-fold stratified CV on 2,000 questions:

- **CV accuracy: 87.25% ± 1.86%**
- **CV macro-F1: 87.02% ± 2.07%**
- Mode 1 precision/recall: 0.92 / 0.95
- Mode 2 precision/recall: 0.95 / 0.86
- Mode 3 precision/recall: 0.85 / **0.99**

**The critical number.** Mode 3 recall = 99%. Of the 335 questions
where Mode 3 was the cheapest correct mode, the router correctly
sent 333 to Mode 3. Two went to Mode 2 (still got correct answers
because Mode 2 was also right on those). **Zero went to Mode 1.**
No safety question was misrouted to LLM-only.

### Step 5.4 — End-to-end accuracy vs compute trade-off

| Strategy | Accuracy | Compute |
|----------|---------:|--------:|
| Always Mode 1 | 40.2% | 1.0× |
| Always Mode 2 | 95.2% | 2.0× |
| Always Mode 3 | 96.6% | 3.0× |
| **Router** | **~96% end-user** | **~1.74×** |
| Oracle | 99.3% | 3.0× |

The router achieves end-user accuracy within 0.5pp of always-Mode-3
while using 42% less compute per question on average (since 40% of
questions are routed to Mode 1, 43% to Mode 2, only 17% to Mode 3).

This is the actual story for the paper: **routing is a compute-efficiency
win, not an accuracy win.** And the safety guarantee (zero Mode 3 →
Mode 1 misroutes) is the headline.

---

## Phase 6 — Production: Streamlit Demo

### Step 6.1 — Why a live demo

Demonstrating the system requires more than CV numbers. A clinician
or evaluator needs to see real questions get routed in real time, see
the retrieved sources, see the citations.

### Step 6.2 — Architecture

**Streamlit single-file app** (`streamlit_app/app.py`) with three layers:

1. **UI layer** — Streamlit components, custom CSS for clinical
   styling (medical-blue primary, light-blue background, hospital-style
   patient banner with allergy alert band).
2. **Pipeline layer** (`lib/pipeline.py`) — `LiveBackend` class loads
   MedCPT, scispaCy, PrimeKG, FAISS indices, MedlinePlus index. Three
   `mode_X` methods each return a record with answer + retrieval
   details.
3. **Router layer** (`lib/router.py`) — `Router` class loads the
   trained MLP weights, takes (question, patient, mode_2_record,
   mode_3_record) and returns predicted mode + probabilities.

### Step 6.3 — Two-button UX

- **"Run"** — uses MLP router to pick one mode and only calls the LLM
  for that mode. Saves compute.
- **"Run all 3 modes"** — runs every mode and shows all three
  side-by-side with the router's pick highlighted. For comparison and
  evaluation.

### Step 6.4 — Citation system

When Mode 2 or Mode 3 fires, the LLM is given numbered references:

- `[1]` Patient prescriptions block
- `[2..]` Patient chart chunks
- `[N..]` MedlinePlus articles (with URLs)
- `[M..]` PrimeKG triples (each triple as one numbered source)

The prompt instructs strict citation rules. After generation, every
`[N]` citation is verified by scoring the surrounding sentence
against the cited source via the MedCPT cross-encoder. Low-relevance
citations are dropped.

A `**Sources**` block is appended to the answer with linked URLs.

### Step 6.5 — Drug-name aliases

Patients ask using brand names ("Tylenol", "Advil", "Vitamin B12").
PrimeKG and MedlinePlus use generic names ("acetaminophen",
"ibuprofen", "cyanocobalamin"). A `_DRUG_ALIASES` dict in
`lib/pipeline.py` maps ~50 common consumer names to KG-canonical
generics, applied in both `kg_for_question` and
`retrieve_medlineplus_chunks`.

### Step 6.6 — Hospital UI styling

- **Background:** light clinical blue (#EFF6FB), white content panels
- **Primary color:** medical blue (#1565C0), not Streamlit's
  default red
- **Patient banner:** EHR-style top strip with name, MRN, demographics,
  and a colored allergy alert band
- **Patient history collapsed by default** — question input is always
  above the fold
- **No emojis** — semantic typography (color, weight, monospace
  for IDs) instead

### Step 6.7 — Project Overview page

A second page in the streamlit nav contains:

- Dataset summary
- Architecture diagram (graphviz: question + EHR → features → router
  → modes → knowledge bases → cited answer)
- Knowledge sources table
- Router architecture (network + 27-feature breakdown)
- Strict-rubric results table
- Safety guarantees

---

## Reflection — What We Learned

### 1. Cross-encoder gating is the most valuable single addition to RAG

Without it, retrieval can hurt more than help. With it, retrieval
becomes the strongest single mode.

Validated twice:
- MedQA Cond5_v2: adaptive injection (9.5% rate) beat blanket
  injection (87.5% rate) by 3.5pp
- Patient portal: cross-encoder threshold prevents low-relevance
  MedlinePlus chunks from being injected into Mode 2

### 2. Adaptive injection beats blanket injection

This insight scaled directly into the router design: don't run KG on
every question, run it only when KG features suggest it'll help.

### 3. A learned router needs separable mode-correctness

Where modes overlap heavily (MedQA: 28% all-correct + 28% all-wrong),
no router can find the signal. Where modes have clear specialization
(patient portal: ~80% of questions have exactly one cheapest correct
mode), the same architecture works.

This was the most important diagnostic of the project: **the router
is only as good as the signal in the data.**

### 4. Prompt engineering can match or beat architectural changes

The Mode 3 diplomatic prompt fix gave +47pp on safety questions. No
new model, no new data, no new pipeline component. Just words.

### 5. Heuristic rubrics need medical scrutiny

The first scoring rubric let Mode 1 win 84% of questions because it
gave credit for non-punt answers that mentioned a drug. The strict
rubric, which disqualifies Mode 1 on safety questions, dropped that
to 40% — and revealed the real distribution that the router needs to
learn.

### 6. Working live demos matter

A streamlit app where you type a question and see the router fire,
the retrieval pull articles, the KG return contraindications, and
the LLM cite [1] [3] inline — communicates the project in 30
seconds. The CV-accuracy table doesn't.

---

## File Map (What Lives Where)

**Streamlit app**
- `streamlit_app/app.py` — UI, page routing, custom CSS
- `streamlit_app/lib/pipeline.py` — `LiveBackend`, all three modes,
  retrieval, KG, MedlinePlus, citation system
- `streamlit_app/lib/router.py` — `RouterMLP`, `build_features()`,
  category regex
- `streamlit_app/.streamlit/config.toml` — theme

**Notebooks (research narrative)**
- `final-project-horus/RAG_MedQA_Gemma2_9b_MEDCPT.ipynb` — baseline
- `final-project-horus/RAG_MedQA_Gemma2_9b_MEDCPT_upgraded.ipynb` —
  6 fixes
- `final-project-horus/PrimeKG_KG_SmokeTest.ipynb` — KG infrastructure
- `final-project-horus/Cond5_v2.ipynb` — smart KG injection
- `final-project-horus/MLP_Router_Training_v2.ipynb` — MedQA router
- `final-project-horus/Patient_RAG_Builder.ipynb` — patient FAISS
- `final-project-horus/Patient_Portal_Pipelines.ipynb` — 3-mode
  pipeline
- `final-project-horus/Mode_3_Prompt_Fix.ipynb` — the +47pp prompt fix
- `final-project-horus/Patient_Portal_Router.ipynb` — patient router
  (23-feature)
- `notebooks/Patient_Portal_A100_Parallel_v2.ipynb` — overnight
  2000-question run

**Scripts (production tooling)**
- `scripts/generate_v2_dataset.py` — 400-patient generator
- `scripts/setup_local_pipeline.py` — downloads MedCPT, builds FAISS
- `scripts/fetch_drug_articles.py` — MedlinePlus scraper
- `scripts/retrain_router_v3.py` — 27-feature retraining (the file
  that produced the live demo's router weights)
- `scripts/rescore_v3_strict.py` — strict medically-aware rubric
- `scripts/make_poster_charts.py` — 5 poster figures

**Data**
- `synthetic_patients/patients_v2.jsonl` — 400 patients
- `synthetic_patients/questions_v2.jsonl` — 1,600 train
- `synthetic_patients/test_questions_v2.jsonl` — 400 test
- `synthetic_patients/patient_portal_responses_v3_2000.jsonl` — 2000
  × 3-mode answers
- `synthetic_patients/patient_portal_ground_truth_v3_2000_STRICT.jsonl`
  — strict labels
- `medlineplus_articles.jsonl` — 2,127 NIH articles

**Trained artifacts**
- `patient_router/router_mlp_v3.pt` — 27-feature MLP weights
- `patient_router/scaler_v3.pkl` — StandardScaler

**Documentation**
- `project_results.md` — full project narrative
- `medqa_experiments_summary.md` — MedQA ablation
- `poster_content.md` — poster sections
- `PROJECT_WALKTHROUGH.md` — this file
- IEEE LaTeX paper (separate file)
