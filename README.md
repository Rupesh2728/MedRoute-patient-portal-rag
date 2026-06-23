# patient-portal-rag

**MedRoute** — a local, context-aware clinical decision-support system that
learns when to retrieve external evidence and when to rely on a large
language model's internal knowledge.

---

## What this is

A patient-facing portal where Gemma2-9B (running locally via Ollama)
answers questions about a patient's prescriptions and conditions. A
learned 27-feature MLP router decides per question whether to use:

- **Mode 1** — LLM only (general knowledge)
- **Mode 2** — LLM + patient EHR retrieval + MedlinePlus citations
- **Mode 3** — LLM + patient EHR + PrimeKG drug-safety triples + MedlinePlus

The router achieves ~96% end-user accuracy while invoking the full RAG +
KG pipeline only on questions that need it (≈42% compute reduction over
always-Mode-3).

## Repository layout

```
streamlit_app/                Live demo (run this)
  app.py                      UI + page routing + CSS
  lib/pipeline.py             LiveBackend, three modes, retrieval, KG, MedlinePlus
  lib/router.py               RouterMLP, build_features, predict_category
  .streamlit/config.toml      Clinical theme

final-project-horus/          Research notebooks (the experimental campaign)
  RAG_MedQA_Gemma2_9b_MEDCPT.ipynb              # baseline
  RAG_MedQA_Gemma2_9b_MEDCPT_upgraded.ipynb     # +6 engineering fixes
  PrimeKG_KG_SmokeTest.ipynb                    # KG coverage validation
  Cond5_v2.ipynb                                # smart KG injection
  MLP_Router_Training_v2.ipynb                  # router on MedQA (failed gracefully)
  Patient_RAG_Builder.ipynb                     # patient FAISS construction
  Patient_Portal_Pipelines.ipynb                # three-mode pipeline
  Mode_3_Prompt_Fix.ipynb                       # +47pp prompt engineering fix
  Patient_Portal_Router.ipynb                   # patient-portal router (23-feature)
  Patient_Portal_TestSet_Eval.ipynb             # held-out eval

notebooks/
  Patient_Portal_A100_Parallel_v2.ipynb         # overnight 2000-question Colab run

scripts/
  generate_v2_dataset.py        Synthetic 400-patient / 2000-question generator
  setup_local_pipeline.py       Downloads MedCPT, builds patient + MedlinePlus FAISS
  fetch_drug_articles.py        MedlinePlus drug catalogue scraper
  retrain_router_v3.py          27-feature router retraining (production)
  rescore_v3_strict.py          Medically-aware strict rubric
  make_poster_charts.py         Chart generators

synthetic_patients/             Data (400 patients, 2000 labelled questions)
medlineplus_articles.jsonl      2,127 NIH articles (15,803 chunks)
patient_router/                 Trained MLP weights + scaler

PROJECT_WALKTHROUGH.md          Step-by-step narrative of every decision
project_results.md              Full results writeup
medqa_experiments_summary.md    MedQA phase ablation
poster_content.md               Poster section text
```

## Quick start (mock mode)

```bash
cd streamlit_app
pip install -r requirements.txt
python -m streamlit run app.py
```

Open http://localhost:8501 and pick a patient. Mock mode shows
pre-computed answers — no GPU or Ollama needed.

## Live mode (real LLM, real retrieval)

1. **Download MedCPT + build FAISS indices** (~10 min, one-time):
   ```bash
   python scripts/setup_local_pipeline.py
   ```
2. **Install Ollama and pull Gemma2** (one-time):
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull gemma2
   ```
3. **Place `primekg_index.pkl`** at the project root (download separately
   from PrimeKG — Harvard precision-medicine knowledge graph).
4. **Toggle "Live mode" in the streamlit sidebar.**

## Headline results

| Strategy | Accuracy (n=2000, strict rubric) | Compute |
|----------|---------------------------------:|--------:|
| Always Mode 1 | 40.2% | 1.0× |
| Always Mode 2 | 95.2% | 2.0× |
| Always Mode 3 | 96.6% | 3.0× |
| **MLP Router** | **~96% end-user** | **~1.74×** |
| Oracle | 99.3% | 3.0× |

Mode 3 recall = 99%: zero safety-critical questions misrouted to LLM-only.

## Authors

Ishaq Miyawala · Maqsood Ahmed · Rupesh Chowdary
University at Buffalo

## License

MIT (see `LICENSE`). All patient data is synthetic.

## Acknowledgements

- **MedCPT** — NCBI / NIH biomedical retrieval encoders
- **PrimeKG** — Harvard Medical School precision-medicine knowledge graph
- **MedlinePlus** — U.S. National Library of Medicine consumer-health corpus
- **scispaCy** — Allen Institute for AI biomedical NLP
- **Gemma 2** — Google DeepMind
- **Ollama** — local LLM serving runtime
