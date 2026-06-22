"""
Live pipeline backend for the Streamlit UI.

Loads MedCPT encoders, patient FAISS index, scispaCy NER, PrimeKG triples,
and exposes three callables that take (question, patient) and return a
mode-result dict matching the JSONL schema produced by the notebooks.

All retrieval logic mirrors `Patient_Portal_Pipelines.ipynb` /
`Mode_3_Prompt_Fix.ipynb` so output is consistent with the offline runs.
"""

from __future__ import annotations

import json
import pickle
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import requests
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Resource paths (override via env vars or constructor)
# ---------------------------------------------------------------------------

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_GEMMA_MODEL = "gemma2"

PREDICATE_PHRASES = {
    "indication": "is indicated for",
    "off-label use": "is used off-label for",
    "contraindication": "is contraindicated in",
    "side effect": "can cause",
    "drug-drug interaction": "interacts with",
    "synergistic interaction": "synergistically interacts with",
    "target": "targets",
    "carrier": "is carried by",
    "transporter": "is transported by",
    "enzyme": "is metabolized by",
    "phenotype present": "presents with",
    "phenotype absent": "is not associated with",
    "associated with": "is associated with",
    "ppi": "interacts with protein",
    "parent-child": "is a type of",
}


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class LiveBackend:
    """
    Holds all live-pipeline state. Initialise once per process; the Streamlit
    app caches the instance so model loads only happen on first render.

    Parameters
    ----------
    project_dir : Path
        Root of the project (the parent of the `streamlit_app/` folder).
    ollama_url, gemma_model : str
        Ollama endpoint and model name.
    """

    def __init__(
        self,
        project_dir: Path,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        gemma_model: str = DEFAULT_GEMMA_MODEL,
    ) -> None:
        self.project_dir = project_dir
        self.ollama_url = ollama_url
        self.gemma_model = gemma_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.query_tokenizer = None
        self.query_encoder = None
        self.cross_tokenizer = None
        self.cross_encoder = None
        self.nlp = None
        self.faiss_index = None
        self.chunk_lookup: dict[str, dict] = {}
        self.patient_chunks_by_id: dict[str, list[dict]] = {}
        self.patient_embeddings_by_id: dict[str, np.ndarray] = {}
        self.name_to_triples: dict = {}

        self.warnings: list[str] = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self) -> None:
        try:
            self._load_encoders()
        except Exception as e:
            self.warnings.append(f"Could not load MedCPT encoders: {e}")

        try:
            self._load_patient_index()
        except Exception as e:
            self.warnings.append(f"Could not load patient FAISS index: {e}")

        try:
            self._load_scispacy()
        except Exception as e:
            self.warnings.append(f"Could not load scispaCy: {e}")

        try:
            self._load_primekg()
        except Exception as e:
            self.warnings.append(f"Could not load PrimeKG: {e}")

    def _load_encoders(self) -> None:
        from transformers import (
            AutoModel,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
        query_path = self.project_dir / "models" / "MedCPT-Query-Encoder"
        cross_path = self.project_dir / "models" / "MedCPT-Cross-Encoder"
        if not query_path.exists() or not cross_path.exists():
            raise FileNotFoundError(
                f"MedCPT encoders missing. Expected at:\n  {query_path}\n  {cross_path}"
            )
        self.query_tokenizer = AutoTokenizer.from_pretrained(str(query_path))
        self.query_encoder = (
            AutoModel.from_pretrained(str(query_path)).to(self.device).eval()
        )
        self.cross_tokenizer = AutoTokenizer.from_pretrained(str(cross_path))
        self.cross_encoder = (
            AutoModelForSequenceClassification.from_pretrained(str(cross_path))
            .to(self.device).eval()
        )

    def _load_patient_index(self) -> None:
        import faiss
        index_path = self.project_dir / "patient_index" / "patient_index.bin"
        chunks_path = self.project_dir / "patient_index" / "patient_chunks.jsonl"
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Patient FAISS missing. Expected:\n  {index_path}\n  {chunks_path}"
            )

        self.faiss_index = faiss.read_index(str(index_path))
        all_chunks: list[dict] = []
        with chunks_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    c = json.loads(line)
                    all_chunks.append(c)
                    self.chunk_lookup[c["chunk_id"]] = c

        all_embeddings = self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal)
        for chunk, emb in zip(all_chunks, all_embeddings):
            pid = chunk["patient_id"]
            self.patient_chunks_by_id.setdefault(pid, []).append(chunk)
            self.patient_embeddings_by_id.setdefault(pid, []).append(emb)
        for pid in self.patient_embeddings_by_id:
            self.patient_embeddings_by_id[pid] = np.array(
                self.patient_embeddings_by_id[pid]
            )

    def _load_scispacy(self) -> None:
        import spacy
        try:
            self.nlp = spacy.load("en_ner_bc5cdr_md")
        except OSError as e:
            raise RuntimeError(
                "scispaCy model 'en_ner_bc5cdr_md' not installed. "
                "Install via the URL in requirements.txt."
            ) from e

    def _load_primekg(self) -> None:
        path = self.project_dir / "primekg_index.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"PrimeKG pickle missing. Expected at: {path}"
            )
        with path.open("rb") as f:
            self.name_to_triples = pickle.load(f)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    @property
    def encoders_ready(self) -> bool:
        return self.query_encoder is not None and self.cross_encoder is not None

    @property
    def patient_index_ready(self) -> bool:
        return self.faiss_index is not None

    @property
    def kg_ready(self) -> bool:
        return bool(self.name_to_triples) and self.nlp is not None

    def status_summary(self) -> dict[str, bool]:
        return {
            "encoders": self.encoders_ready,
            "patient_index": self.patient_index_ready,
            "kg": self.kg_ready,
            "ollama": self._ping_ollama(),
        }

    def _ping_ollama(self) -> bool:
        try:
            base = self.ollama_url.replace("/api/generate", "/api/tags")
            r = requests.get(base, timeout=2)
            return r.ok
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def _encode_query(self, text: str) -> np.ndarray:
        encoded = self.query_tokenizer(
            [text], truncation=True, padding=True,
            max_length=64, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            out = self.query_encoder(**encoded)
            emb = out.last_hidden_state[:, 0, :]
            emb = F.normalize(emb, dim=-1)
        return emb.cpu().numpy()

    def retrieve_patient_chunks(
        self, question: str, patient_id: str, top_k: int = 4
    ) -> list[dict]:
        if not self.encoders_ready or patient_id not in self.patient_embeddings_by_id:
            return []
        qe = self._encode_query(question)
        emb = self.patient_embeddings_by_id[patient_id]
        chunks = self.patient_chunks_by_id[patient_id]
        scores = (emb @ qe.T).flatten()
        ranked = np.argsort(-scores)[:top_k]
        return [
            {
                "score": float(scores[i]),
                "chunk_id": chunks[i]["chunk_id"],
                "section": chunks[i]["section"],
                "text": chunks[i]["text"],
            }
            for i in ranked
        ]

    def rerank_chunks(
        self, question: str, candidates: list[dict], top_k: int = 3
    ) -> list[dict]:
        if not self.encoders_ready or not candidates:
            return candidates[:top_k]
        pairs = [[question, c["text"]] for c in candidates]
        scores: list[float] = []
        batch_size = 8
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            encoded = self.cross_tokenizer(
                batch, truncation=True, padding=True,
                return_tensors="pt", max_length=512,
            ).to(self.device)
            with torch.no_grad():
                logits = self.cross_encoder(**encoded).logits.squeeze(dim=-1)
            if logits.dim() == 0:
                scores.append(float(logits))
            else:
                scores.extend(logits.cpu().tolist())
        for c, s in zip(candidates, scores):
            c["cross_encoder_score"] = float(s)
        return sorted(
            candidates, key=lambda x: x["cross_encoder_score"], reverse=True
        )[:top_k]

    # ------------------------------------------------------------------
    # KG
    # ------------------------------------------------------------------
    def kg_for_drugs(self, drug_names: list[str], top_k_per_drug: int = 4) -> list[dict]:
        if not self.kg_ready:
            return []
        triples: list[dict] = []
        seen: set[tuple] = set()
        for drug in drug_names:
            for variant in [drug.lower(), drug.lower().split()[0]]:
                if variant in self.name_to_triples:
                    for rel, y_name, y_type in self.name_to_triples[variant][:top_k_per_drug]:
                        key = (variant, rel, y_name)
                        if key in seen:
                            continue
                        seen.add(key)
                        triples.append({
                            "subject": variant,
                            "predicate": rel,
                            "object": y_name,
                            "object_type": y_type,
                            "triple_text": f"{variant} {rel} {y_name}",
                        })
                    break
        return triples

    def kg_for_question(self, question: str, top_k: int = 3) -> list[dict]:
        if not self.kg_ready:
            return []
        doc = self.nlp(question)
        triples: list[dict] = []
        seen: set[tuple] = set()
        for ent in doc.ents:
            e = ent.text.strip().lower()
            if e in self.name_to_triples:
                for rel, y_name, y_type in self.name_to_triples[e][:top_k]:
                    key = (e, rel, y_name)
                    if key in seen:
                        continue
                    seen.add(key)
                    triples.append({
                        "subject": e,
                        "predicate": rel,
                        "object": y_name,
                        "object_type": y_type,
                        "triple_text": f"{e} {rel} {y_name}",
                    })
        return triples

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------
    @staticmethod
    def _triple_to_sentence(t: dict) -> str:
        subj = t["subject"].capitalize()
        pred = t["predicate"].lower()
        obj = t["object"]
        phrase = PREDICATE_PHRASES.get(pred, pred)
        return f"{subj} {phrase} {obj}."

    @staticmethod
    def _format_prescription_block(patient: dict) -> str:
        meds = patient.get("medications", [])
        if not meds:
            return "(no active prescriptions on file)"
        lines = []
        for m in meds:
            line = f"- {m['drug']} {m['dosage']}, {m['frequency']}, for {m['indication']}"
            if m.get("notes"):
                line += f" (Note: {m['notes']})"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _format_patient_context(reranked: list[dict]) -> str:
        if not reranked:
            return ""
        return "=== PATIENT MEDICAL RECORD ===\n" + "\n".join(
            f"- {c['text']}" for c in reranked
        )

    def _format_kg_block(self, triples: list[dict]) -> str:
        if not triples:
            return ""
        return (
            "=== DRUG SAFETY INFORMATION (from validated drug databases) ===\n"
            + "\n".join(f"- {self._triple_to_sentence(t)}" for t in triples)
        )

    def _build_prompt_v1(
        self, question: str, patient: dict,
        patient_context: str = "", kg_block: str = "",
    ) -> str:
        rx = self._format_prescription_block(patient)
        parts = [
            "You are a helpful medical assistant answering questions for a patient about their",
            "prescriptions and health. Use the information provided to give a clear, accurate,",
            "and patient-friendly answer. If you do not have the information needed, say so",
            "honestly rather than guessing.",
            "",
            "=== ACTIVE PRESCRIPTIONS ===",
            rx, "",
        ]
        if patient_context:
            parts += [patient_context, ""]
        if kg_block:
            parts += [kg_block, ""]
        parts += ["=== PATIENT QUESTION ===", question, "", "Provide a concise, helpful answer:"]
        return "\n".join(parts)

    def _build_prompt_mode3(
        self, question: str, patient: dict,
        patient_context: str, kg_block: str,
    ) -> str:
        rx = self._format_prescription_block(patient)
        return (
            "You are a helpful medical assistant answering questions for a patient about their\n"
            "prescriptions and health. Use the patient's prescriptions, medical record, and the\n"
            "drug safety information below to give a thoughtful, helpful answer.\n"
            "\n"
            "If the safety information directly addresses the question, share it in plain language.\n"
            "Recommending the patient confirm with their doctor is appropriate, but try to be\n"
            "informative first rather than only deferring. When the medical facts above clearly\n"
            "answer the question, lead with that information.\n"
            "\n"
            f"=== ACTIVE PRESCRIPTIONS ===\n{rx}\n"
            f"\n{patient_context}\n"
            f"\n{kg_block}\n"
            "\n=== PATIENT QUESTION ===\n"
            f"{question}\n"
            "\nProvide a concise, helpful answer:"
        )

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    def query_gemma(
        self, prompt: str, temperature: float = 0.0,
        max_tokens: int = 512, seed: int = 42,
    ) -> str:
        payload = {
            "model": self.gemma_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "seed": seed,
            },
        }
        try:
            resp = requests.post(self.ollama_url, json=payload, timeout=180)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            return f"[Ollama error: {e}]"

    # ------------------------------------------------------------------
    # Modes
    # ------------------------------------------------------------------
    def mode_1(self, question: str, patient: dict) -> dict:
        t0 = time.time()
        prompt = self._build_prompt_v1(question, patient)
        answer = self.query_gemma(prompt)
        return {
            "mode": 1, "mode_name": "LLM_only",
            "answer": answer,
            "latency_seconds": round(time.time() - t0, 2),
            "n_retrieved_chunks": 0, "n_kg_triples": 0,
            "retrieved_chunks": [], "kg_triples": [],
        }

    def mode_2(self, question: str, patient: dict) -> dict:
        t0 = time.time()
        candidates = self.retrieve_patient_chunks(
            question, patient["patient_id"], top_k=4
        )
        reranked = self.rerank_chunks(question, candidates, top_k=3)
        ctx = self._format_patient_context(reranked)
        prompt = self._build_prompt_v1(question, patient, patient_context=ctx)
        answer = self.query_gemma(prompt)
        return {
            "mode": 2, "mode_name": "RAG",
            "answer": answer,
            "latency_seconds": round(time.time() - t0, 2),
            "n_retrieved_chunks": len(reranked), "n_kg_triples": 0,
            "retrieved_chunks": [
                {
                    "section": c["section"],
                    "score": c.get("cross_encoder_score", c["score"]),
                    "text": c["text"][:200],
                }
                for c in reranked
            ],
            "kg_triples": [],
        }

    def mode_3(self, question: str, patient: dict) -> dict:
        t0 = time.time()
        candidates = self.retrieve_patient_chunks(
            question, patient["patient_id"], top_k=4
        )
        reranked = self.rerank_chunks(question, candidates, top_k=3)
        ctx = self._format_patient_context(reranked)

        drug_names = [m["drug"] for m in patient.get("medications", [])]
        drug_triples = self.kg_for_drugs(drug_names, top_k_per_drug=4)
        question_triples = self.kg_for_question(question, top_k=3)

        seen: set[tuple] = set()
        all_triples: list[dict] = []
        for t in drug_triples + question_triples:
            key = (t["subject"], t["predicate"], t["object"])
            if key in seen:
                continue
            seen.add(key)
            all_triples.append(t)

        kg_block = self._format_kg_block(all_triples)
        prompt = self._build_prompt_mode3(question, patient, ctx, kg_block)
        answer = self.query_gemma(prompt)

        return {
            "mode": 3, "mode_name": "RAG_KG",
            "answer": answer,
            "latency_seconds": round(time.time() - t0, 2),
            "n_retrieved_chunks": len(reranked),
            "n_kg_triples": len(all_triples),
            "retrieved_chunks": [
                {
                    "section": c["section"],
                    "score": c.get("cross_encoder_score", c["score"]),
                    "text": c["text"][:200],
                }
                for c in reranked
            ],
            "kg_triples": [t["triple_text"] for t in all_triples],
        }
