# Hybrid RAG Patient Portal — Streamlit Demo

Two-mode demo UI for the patient portal pipeline.

- **Mock mode** (default): no LLM needed, shows pre-computed responses from the JSONL files. Demo-safe; works on any laptop in 5 seconds.
- **Live mode** (toggle in sidebar): loads MedCPT, scispaCy, PrimeKG, and the trained MLP router; calls Gemma2 via Ollama in real time for each question typed.

## Quick start (Mock mode only)

```bash
cd streamlit_app
pip install streamlit
streamlit run app.py
```

Open http://localhost:8501 → toggle "Live mode" off in the sidebar.

## Live mode setup

You need:

1. **Ollama running** with `gemma2` pulled:
   ```bash
   ollama serve            # in one terminal
   ollama pull gemma2      # one-time, ~5 GB
   ```

2. **Python deps**:
   ```bash
   pip install -r requirements.txt
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
   ```

3. **Project artifacts** in the project root (parent of `streamlit_app/`):

   ```
   patient_index/
     patient_index.bin
     patient_chunks.jsonl
     patient_chunk_ids.json

   models/
     MedCPT-Query-Encoder/      (HF cache; produced by Patient_RAG_Builder.ipynb)
     MedCPT-Cross-Encoder/

   primekg_index.pkl

   patient_router/
     router_mlp.pt
     scaler.pkl
   ```

   Copy these from your Drive (`/content/drive/MyDrive/DL Project/...`) to the local project root.

4. **Synthetic patients data** in `synthetic_patients/` (already there if you've run the notebooks).

Then:

```bash
streamlit run app.py
```

Toggle **Live mode** in the sidebar. First question takes ~30 s to warm up encoders + Ollama; subsequent calls are ~2–5 s.

## What's in the UI

### Patient Portal page

- Sidebar: pick any of the 50 synthetic patients
- Main panel: patient demographics, conditions, allergies, vitals, lifestyle, prescriptions
- Question entry:
  - **Mock mode**: dropdown of 350 pre-computed questions (250 train + 100 test, tagged)
  - **Live mode**: free-text input, runs all 3 modes via real pipeline
- Router decision: predicted mode, confidence, mode probabilities, expected mode (when known)
- Three response columns side-by-side (Mode 1 / 2 / 3) with:
  - Live or saved answer
  - ✓/✗ correctness badge (mock mode only)
  - "ROUTER PICK" highlight on the chosen column
  - Expandable retrieval / KG details

### Project Overview page

- Stats and architecture diagram
- Train vs test result table

## Notes

- All patient data is synthetic.
- Live mode uses CPU-friendly faiss; GPU is detected automatically when available.
- If `numpy` or `scispacy` install conflicts, restart Python after installing them (Streamlit must be restarted too).
