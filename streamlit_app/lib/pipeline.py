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

        # MedlinePlus knowledge index (for Mode 2 citations)
        self.medline_index = None
        self.medline_chunks: list[dict] = []
        self.medline_embeddings: np.ndarray | None = None

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

        try:
            self._load_medlineplus()
        except Exception as e:
            self.warnings.append(f"Could not load MedlinePlus: {e}")

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

    def _load_medlineplus(self) -> None:
        import faiss
        idx_path = self.project_dir / "medlineplus_index" / "medlineplus_index.bin"
        chunks_path = self.project_dir / "medlineplus_index" / "medlineplus_chunks.jsonl"
        if not idx_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"MedlinePlus index missing. Expected:\n  {idx_path}\n  {chunks_path}"
            )
        self.medline_index = faiss.read_index(str(idx_path))
        with chunks_path.open() as f:
            self.medline_chunks = [json.loads(line) for line in f if line.strip()]
        self.medline_embeddings = self.medline_index.reconstruct_n(
            0, self.medline_index.ntotal
        )

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

    @property
    def medline_ready(self) -> bool:
        return self.medline_index is not None and bool(self.medline_chunks)

    def status_summary(self) -> dict[str, bool]:
        return {
            "encoders": self.encoders_ready,
            "patient_index": self.patient_index_ready,
            "medlineplus": self.medline_ready,
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

    def retrieve_medlineplus_chunks(
        self, question: str, top_k: int = 4, patient: dict | None = None,
    ) -> list[dict]:
        """Retrieve top-K MedlinePlus chunks, filtered to drug-relevant articles.

        Strategy:
          1. Build target_drugs = patient's prescribed drugs ∪ drugs detected
             in the question (via scispaCy NER + substring scan).
          2. Filter MedlinePlus chunks to those whose `matched_terms` intersect
             target_drugs.
          3. Within that filtered pool, rank by MedCPT similarity to question.
          4. Fall back to global question-based retrieval only if no
             drug-relevant chunks exist.
        """
        if not self.encoders_ready or not self.medline_ready:
            return []

        # Build target drug set
        target_drugs: set[str] = set()
        if patient is not None:
            for med in patient.get("medications", []):
                target_drugs.add(med["drug"].split()[0].lower())

        # Add drugs detected in the question via scispaCy
        if self.nlp is not None:
            try:
                for ent in self.nlp(question).ents:
                    target_drugs.add(ent.text.strip().lower())
            except Exception:
                pass

        # Filter chunks to those whose article was about a target drug
        candidate_idxs: list[int] = []
        if target_drugs:
            for i, c in enumerate(self.medline_chunks):
                mt = {t.lower() for t in c.get("matched_terms", [])}
                # match if any target drug appears in matched_terms
                # (use substring on either direction to be lenient)
                if mt & target_drugs or any(
                    any(td in t or t in td for t in mt) for td in target_drugs
                ):
                    candidate_idxs.append(i)

        qe = self._encode_query(question)

        if candidate_idxs:
            cand_embs = self.medline_embeddings[candidate_idxs]
            scores = (cand_embs @ qe.T).flatten()
            top = np.argsort(-scores)[:top_k]
            return [
                {
                    "score": float(scores[t]),
                    "chunk_id": self.medline_chunks[candidate_idxs[t]]["chunk_id"],
                    "title": self.medline_chunks[candidate_idxs[t]]["title"],
                    "url": self.medline_chunks[candidate_idxs[t]]["url"],
                    "topic_type": self.medline_chunks[candidate_idxs[t]].get("topic_type", "topic"),
                    "text": self.medline_chunks[candidate_idxs[t]]["text"],
                    "source_kind": "medlineplus",
                }
                for t in top
            ]

        # Fallback: question-similarity over all chunks (no drug match found)
        scores = (self.medline_embeddings @ qe.T).flatten()
        top = np.argsort(-scores)[:top_k]
        return [
            {
                "score": float(scores[i]),
                "chunk_id": self.medline_chunks[i]["chunk_id"],
                "title": self.medline_chunks[i]["title"],
                "url": self.medline_chunks[i]["url"],
                "topic_type": self.medline_chunks[i].get("topic_type", "topic"),
                "text": self.medline_chunks[i]["text"],
                "source_kind": "medlineplus",
            }
            for i in top
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
    # Consumer-name → generic-name aliases. PrimeKG uses generic names
    # (cyanocobalamin, acetaminophen) but patients ask using brand or
    # consumer names (Tylenol, Vitamin B12). Map both directions on lookup.
    _DRUG_ALIASES: dict[str, list[str]] = {
        # OTC pain relievers
        "tylenol": ["acetaminophen", "paracetamol"],
        "advil": ["ibuprofen"],
        "motrin": ["ibuprofen"],
        "aleve": ["naproxen"],
        "naprosyn": ["naproxen"],
        "bayer": ["aspirin"],
        "ecotrin": ["aspirin"],
        # Vitamins / supplements
        "vitamin b12": ["cyanocobalamin", "methylcobalamin", "cobalamin"],
        "b12": ["cyanocobalamin", "methylcobalamin", "cobalamin"],
        "cobalamin": ["cyanocobalamin"],
        "vitamin d": ["cholecalciferol", "ergocalciferol"],
        "vitamin d3": ["cholecalciferol"],
        "vitamin c": ["ascorbic acid"],
        "vitamin b6": ["pyridoxine"],
        "vitamin b1": ["thiamine"],
        "vitamin b2": ["riboflavin"],
        "vitamin b9": ["folic acid", "folate"],
        "folate": ["folic acid"],
        "fish oil": ["omega-3 fatty acids", "docosahexaenoic acid", "eicosapentaenoic acid"],
        "calcium": ["calcium carbonate", "calcium citrate"],
        "iron": ["ferrous sulfate", "ferrous fumarate", "ferrous gluconate"],
        "magnesium": ["magnesium oxide", "magnesium citrate"],
        "potassium": ["potassium chloride"],
        # Common antihistamines / cold meds
        "benadryl": ["diphenhydramine"],
        "claritin": ["loratadine"],
        "zyrtec": ["cetirizine"],
        "allegra": ["fexofenadine"],
        "sudafed": ["pseudoephedrine"],
        "mucinex": ["guaifenesin"],
        # GI / antacids
        "tums": ["calcium carbonate"],
        "pepto": ["bismuth subsalicylate"],
        "imodium": ["loperamide"],
        "prilosec": ["omeprazole"],
        "nexium": ["esomeprazole"],
        "zantac": ["ranitidine", "famotidine"],
        # Cardio brand names
        "zestril": ["lisinopril"],
        "prinivil": ["lisinopril"],
        "norvasc": ["amlodipine"],
        "lipitor": ["atorvastatin"],
        "crestor": ["rosuvastatin"],
        "zocor": ["simvastatin"],
        "coumadin": ["warfarin"],
        "plavix": ["clopidogrel"],
        # Diabetes brand names
        "glucophage": ["metformin"],
        "januvia": ["sitagliptin"],
        "jardiance": ["empagliflozin"],
        "ozempic": ["semaglutide"],
        # Lifestyle substances
        "alcohol": ["ethanol"],
        "weed": ["cannabis", "tetrahydrocannabinol"],
        "marijuana": ["cannabis", "tetrahydrocannabinol"],
        "caffeine": ["caffeine"],
        "coffee": ["caffeine"],
    }

    def _expand_drug_name(self, name: str) -> list[str]:
        """Return name + any KG-canonical aliases for it."""
        n = name.strip().lower()
        out = [n]
        if n in self._DRUG_ALIASES:
            out.extend(self._DRUG_ALIASES[n])
        return out

    # Predicate priority for safety-relevant questions: drug-drug interactions
    # and side effects come first; biological targets/transporters last.
    _SAFETY_PRIORITY = {
        "drug-drug interaction": 0,
        "synergistic interaction": 0,
        "contraindication": 0,
        "side effect": 1,
        "indication": 2,
        "off-label use": 2,
        "associated with": 3,
    }

    def _predicate_priority(self, rel: str) -> int:
        return self._SAFETY_PRIORITY.get(rel.lower(), 9)

    def kg_for_drugs(self, drug_names: list[str], top_k_per_drug: int = 6,
                      prioritize_safety: bool = True) -> list[dict]:
        if not self.kg_ready:
            return []
        triples: list[dict] = []
        seen: set[tuple] = set()
        for drug in drug_names:
            base_variants = [drug.lower(), drug.lower().split()[0]]
            # Expand each variant via the alias table (e.g. brand → generic)
            all_variants: list[str] = []
            for v in base_variants:
                all_variants.extend(self._expand_drug_name(v))
            for variant in all_variants:
                if variant in self.name_to_triples:
                    all_for_drug = self.name_to_triples[variant]
                    if prioritize_safety:
                        # Sort by predicate priority so interactions / side
                        # effects bubble to the top of each drug's list.
                        all_for_drug = sorted(
                            all_for_drug,
                            key=lambda t: self._predicate_priority(t[0]),
                        )
                    for rel, y_name, y_type in all_for_drug[:top_k_per_drug]:
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

    def kg_for_question(self, question: str, top_k: int = 3,
                        max_total: int = 20) -> list[dict]:
        """
        Pull KG triples for entities the question mentions.

        Three lookup strategies, in order:
        1. scispaCy-extracted entities (full entity text, lowered)
        2. Variant fallback for each entity (first word, hyphen-split)
        3. Brute-force keyword scan over question words (catches brand names
           and anything scispaCy missed)
        """
        if not self.kg_ready:
            return []
        triples: list[dict] = []
        seen_triples: set[tuple] = set()
        tried_subjects: set[str] = set()

        def add_for_subject(subj: str) -> None:
            if subj in tried_subjects or len(triples) >= max_total:
                return
            tried_subjects.add(subj)
            if subj not in self.name_to_triples:
                return
            # Apply safety-priority sort so interactions / contraindications
            # / side effects come before biological targets/transporters.
            sorted_triples = sorted(
                self.name_to_triples[subj],
                key=lambda t: self._predicate_priority(t[0]),
            )
            for rel, y_name, y_type in sorted_triples[:top_k]:
                key = (subj, rel, y_name)
                if key in seen_triples:
                    continue
                seen_triples.add(key)
                triples.append({
                    "subject": subj, "predicate": rel, "object": y_name,
                    "object_type": y_type,
                    "triple_text": f"{subj} {rel} {y_name}",
                })
                if len(triples) >= max_total:
                    return

        # 1+2: scispaCy entities with variant + alias fallback
        doc = self.nlp(question)
        for ent in doc.ents:
            e = ent.text.strip().lower()
            base_variants = [e]
            if e.split():
                base_variants.append(e.split()[0])
            if e:
                base_variants.append(e.replace("-", " ").split()[0])
            # Expand each variant via the alias table (e.g. "vitamin b12" → ["cyanocobalamin"])
            for variant in base_variants:
                if not variant:
                    continue
                for alias in self._expand_drug_name(variant):
                    add_for_subject(alias)

        # 3: Brute-force keyword scan over question words (length > 4) plus
        # known multi-word aliases (e.g. "vitamin b12", "fish oil").
        q_lower = question.lower()
        for word in re.findall(r"\b[a-z]{5,}\b", q_lower):
            for alias in self._expand_drug_name(word):
                add_for_subject(alias)

        # Multi-word alias scan: check for any alias key present in question
        for alias_key in self._DRUG_ALIASES:
            if " " in alias_key and alias_key in q_lower:
                for alias in self._expand_drug_name(alias_key):
                    add_for_subject(alias)

        # Special case: bare alphanumeric drug tokens like "b12", "d3"
        for tok in re.findall(r"\b[a-z]\d+\b", q_lower):
            for alias in self._expand_drug_name(tok):
                add_for_subject(alias)

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

    def _build_sources_mode3(
        self, patient_chunks: list[dict], medline_chunks: list[dict],
        kg_triples: list[dict], patient: dict | None = None,
    ) -> list[dict]:
        """Numbered sources for Mode 3: prescriptions → patient chart →
        MedlinePlus → KG triples (each triple is one numbered source)."""
        sources: list[dict] = []
        sid = 1
        if patient and patient.get("medications"):
            sources.append({
                "id": sid, "label": "Patient prescriptions",
                "text": self._format_prescription_block(patient),
                "kind": "prescriptions", "url": "",
            })
            sid += 1
        for c in patient_chunks:
            sources.append({
                "id": sid,
                "label": f"Patient record ({c['section'].replace('_', ' ')})",
                "text": c["text"], "kind": "patient", "url": "",
            })
            sid += 1
        for c in medline_chunks:
            sources.append({
                "id": sid, "label": f"MedlinePlus: {c['title']}",
                "text": c["text"], "kind": "medlineplus",
                "url": c.get("url", ""),
            })
            sid += 1
        for t in kg_triples:
            sentence = self._triple_to_sentence(t)
            sources.append({
                "id": sid,
                "label": f"PrimeKG: {t['subject']} {t['predicate']} {t['object']}",
                "text": sentence, "kind": "kg", "url": "",
            })
            sid += 1
        return sources

    def _build_prompt_mode3_with_citations(
        self, question: str, patient: dict, sources: list[dict],
    ) -> str:
        rx = self._format_prescription_block(patient)
        ref_block = self._format_cited_context(sources)
        return (
            "You are a helpful medical assistant answering questions for a patient\n"
            "about their prescriptions and health. The numbered references below are\n"
            "the ONLY things you may cite. Drug safety information from validated\n"
            "biomedical databases (PrimeKG) and authoritative consumer-health sources\n"
            "(MedlinePlus) is included. Lead with concrete information when the\n"
            "references support it; deferring to a doctor is appropriate only as a\n"
            "secondary recommendation.\n"
            "\n"
            "Citation rules (strict):\n"
            "1. After each factual claim, add a citation like [1] or [3].\n"
            "2. Cite a reference ONLY if its text directly supports the claim.\n"
            "3. Do not invent reference numbers. Do not cite [N] for an unrelated topic.\n"
            "4. You may cite multiple references for one claim ([1][3]).\n"
            "\n"
            f"=== ACTIVE PRESCRIPTIONS ===\n{rx}\n"
            f"\n{ref_block}\n"
            "\n=== PATIENT QUESTION ===\n"
            f"{question}\n"
            "\nProvide a concise, helpful answer with inline [N] citations:"
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
    def prepare_features(
        self, question: str, patient: dict,
    ) -> tuple[list[dict], list[dict], dict, dict]:
        """
        Run patient retrieval + MedlinePlus retrieval + KG lookup (no LLM call).
        Returns the raw artefacts plus partial mode_2 / mode_3 records the
        router needs for features.
        """
        # Patient chunks
        candidates = self.retrieve_patient_chunks(
            question, patient["patient_id"], top_k=4
        )
        reranked = self.rerank_chunks(question, candidates, top_k=3)

        # MedlinePlus chunks (for router features)
        cand_m = self.retrieve_medlineplus_chunks(question, top_k=10, patient=patient)
        reranked_m_all = (
            self.rerank_chunks(question, cand_m, top_k=10) if cand_m else []
        )
        # Keep all reranked MedlinePlus for feature extraction (no threshold)
        medline_top = reranked_m_all[:5]

        # KG triples
        drug_names = [m["drug"] for m in patient.get("medications", [])]
        drug_triples = self.kg_for_drugs(drug_names, top_k_per_drug=6)
        question_triples = self.kg_for_question(question, top_k=5)
        seen: set[tuple] = set()
        all_triples: list[dict] = []
        for t in drug_triples + question_triples:
            key = (t["subject"], t["predicate"], t["object"])
            if key in seen:
                continue
            seen.add(key)
            all_triples.append(t)

        mode_2_record = {
            "retrieved_chunks": [
                {
                    "section": c["section"],
                    "score": c.get("cross_encoder_score", c["score"]),
                    "text": c["text"][:200],
                }
                for c in reranked
            ],
            "n_retrieved_chunks": len(reranked),
            # NEW: MedlinePlus features for the router
            "medlineplus_chunks": [
                {
                    "title": c["title"], "url": c.get("url", ""),
                    "score": c.get("cross_encoder_score", c["score"]),
                    "text": c["text"][:200],
                }
                for c in medline_top
            ],
            "n_medlineplus_chunks": len(medline_top),
        }
        mode_3_record = {
            "n_kg_triples": len(all_triples),
            "kg_triples": [t["triple_text"] for t in all_triples],
        }
        return reranked, all_triples, mode_2_record, mode_3_record

    def answer_for_mode(
        self, mode: int, question: str, patient: dict,
        reranked: list[dict], all_triples: list[dict],
    ) -> dict:
        """LLM call for a single mode using already-computed retrieval + KG."""
        t0 = time.time()
        if mode == 1:
            prompt = self._build_prompt_v1(question, patient)
            answer = self.query_gemma(prompt)
            return {
                "mode": 1, "mode_name": "LLM_only",
                "answer": answer,
                "latency_seconds": round(time.time() - t0, 2),
                "n_retrieved_chunks": 0, "n_kg_triples": 0,
                "retrieved_chunks": [], "kg_triples": [],
            }
        if mode == 2:
            for c in reranked:
                c.setdefault("source_kind", "patient")
            cand_m = self.retrieve_medlineplus_chunks(question, top_k=10, patient=patient)
            reranked_m_all = self.rerank_chunks(question, cand_m, top_k=10) if cand_m else []
            reranked_m = [
                c for c in reranked_m_all
                if c.get("cross_encoder_score", c.get("score", 0)) > -3.0
            ][:2]
            sources = self._build_sources(reranked, reranked_m, patient)
            ctx = self._format_cited_context(sources)
            prompt = self._build_prompt_with_citations(question, patient, ctx)
            answer = self.query_gemma(prompt)
            answer = self._verify_citations(answer, sources, threshold=-1.0)
            answer_with_refs = self._append_sources_block(answer, sources)
            return {
                "mode": 2, "mode_name": "RAG",
                "answer": answer_with_refs,
                "latency_seconds": round(time.time() - t0, 2),
                "n_retrieved_chunks": len(reranked),
                "n_medlineplus_chunks": len(reranked_m),
                "n_kg_triples": 0,
                "retrieved_chunks": [
                    {
                        "section": c["section"],
                        "score": c.get("cross_encoder_score", c["score"]),
                        "text": c["text"][:200],
                    }
                    for c in reranked
                ],
                "medlineplus_chunks": [
                    {
                        "title": c["title"], "url": c["url"],
                        "score": c.get("cross_encoder_score", c["score"]),
                        "text": c["text"][:200],
                    }
                    for c in reranked_m
                ],
                "sources": [
                    {"id": s["id"], "label": s["label"], "url": s.get("url", "")}
                    for s in sources
                ],
                "kg_triples": [],
            }
        if mode == 3:
            # Pull MedlinePlus chunks for patient drugs + question drugs
            cand_m = self.retrieve_medlineplus_chunks(
                question, top_k=10, patient=patient
            )
            reranked_m_all = (
                self.rerank_chunks(question, cand_m, top_k=10)
                if cand_m else []
            )
            reranked_m = [
                c for c in reranked_m_all
                if c.get("cross_encoder_score", c.get("score", 0)) > -3.0
            ][:2]

            # Build numbered sources + citation-aware Mode 3 prompt
            sources = self._build_sources_mode3(
                reranked, reranked_m, all_triples, patient
            )
            prompt = self._build_prompt_mode3_with_citations(
                question, patient, sources
            )
            answer = self.query_gemma(prompt)
            answer = self._verify_citations(answer, sources, threshold=-1.0)
            answer_with_refs = self._append_sources_block(answer, sources)

            return {
                "mode": 3, "mode_name": "RAG_KG",
                "answer": answer_with_refs,
                "latency_seconds": round(time.time() - t0, 2),
                "n_retrieved_chunks": len(reranked),
                "n_medlineplus_chunks": len(reranked_m),
                "n_kg_triples": len(all_triples),
                "retrieved_chunks": [
                    {
                        "section": c["section"],
                        "score": c.get("cross_encoder_score", c["score"]),
                        "text": c["text"][:200],
                    }
                    for c in reranked
                ],
                "medlineplus_chunks": [
                    {
                        "title": c["title"], "url": c.get("url", ""),
                        "score": c.get("cross_encoder_score", c["score"]),
                        "text": c["text"][:200],
                    }
                    for c in reranked_m
                ],
                "kg_triples": [t["triple_text"] for t in all_triples],
                "sources": [
                    {"id": s["id"], "label": s["label"], "url": s.get("url", "")}
                    for s in sources
                ],
            }
        raise ValueError(f"Unknown mode: {mode}")

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
        # Patient chunks: retrieve wider (top-6), rerank, threshold-filter
        cand_p = self.retrieve_patient_chunks(
            question, patient["patient_id"], top_k=6
        )
        reranked_p_all = self.rerank_chunks(question, cand_p, top_k=6)
        # Keep top-3 patient chunks regardless of score (chart context is always
        # somewhat useful), but flag low-confidence ones
        reranked_p = reranked_p_all[:3]
        for c in reranked_p:
            c["source_kind"] = "patient"

        # MedlinePlus chunks: wider initial retrieval (top-10), rerank,
        # threshold-filter (only keep chunks with cross-encoder score > 0)
        cand_m = self.retrieve_medlineplus_chunks(question, top_k=10, patient=patient)
        reranked_m_all = self.rerank_chunks(question, cand_m, top_k=10) if cand_m else []
        MEDLINE_SCORE_THRESHOLD = 0.0
        reranked_m = [
            c for c in reranked_m_all
            if c.get("cross_encoder_score", c.get("score", 0)) > MEDLINE_SCORE_THRESHOLD
        ][:2]

        # Build numbered sources + citation-aware context
        sources = self._build_sources(reranked_p, reranked_m, patient)
        ctx = self._format_cited_context(sources)
        prompt = self._build_prompt_with_citations(question, patient, ctx)
        answer = self.query_gemma(prompt)
        answer_with_refs = self._append_sources_block(answer, sources)

        return {
            "mode": 2, "mode_name": "RAG",
            "answer": answer_with_refs,
            "latency_seconds": round(time.time() - t0, 2),
            "n_retrieved_chunks": len(reranked_p),
            "n_medlineplus_chunks": len(reranked_m),
            "n_kg_triples": 0,
            "retrieved_chunks": [
                {
                    "section": c["section"],
                    "score": c.get("cross_encoder_score", c["score"]),
                    "text": c["text"][:200],
                }
                for c in reranked_p
            ],
            "medlineplus_chunks": [
                {
                    "title": c["title"],
                    "url": c["url"],
                    "score": c.get("cross_encoder_score", c["score"]),
                    "text": c["text"][:200],
                }
                for c in reranked_m
            ],
            "sources": [
                {"id": s["id"], "label": s["label"], "url": s.get("url", "")}
                for s in sources
            ],
            "kg_triples": [],
        }

    # ------------------------------------------------------------------
    # Citation helpers
    # ------------------------------------------------------------------
    def _build_sources(
        self, patient_chunks: list[dict], medline_chunks: list[dict],
        patient: dict | None = None,
    ) -> list[dict]:
        """Numbered source list: [1] = active prescriptions block,
        [2..N] = patient chart chunks, [N+1..] = MedlinePlus chunks."""
        sources: list[dict] = []
        sid = 1
        # [1] Active prescriptions (always first if patient has any)
        if patient and patient.get("medications"):
            rx = self._format_prescription_block(patient)
            sources.append({
                "id": sid, "label": "Patient prescriptions",
                "text": rx, "kind": "prescriptions", "url": "",
            })
            sid += 1
        for c in patient_chunks:
            label = f"Patient record ({c['section'].replace('_', ' ')})"
            sources.append({
                "id": sid, "label": label, "text": c["text"],
                "kind": "patient", "url": "",
            })
            sid += 1
        for c in medline_chunks:
            label = f"MedlinePlus: {c['title']}"
            sources.append({
                "id": sid, "label": label, "text": c["text"],
                "kind": "medlineplus", "url": c.get("url", ""),
            })
            sid += 1
        return sources

    @staticmethod
    def _format_cited_context(sources: list[dict]) -> str:
        if not sources:
            return ""
        parts = ["=== REFERENCE MATERIAL (cite inline as [N]) ==="]
        for s in sources:
            parts.append(f"[{s['id']}] {s['label']}\n{s['text']}")
        return "\n\n".join(parts)

    def _build_prompt_with_citations(
        self, question: str, patient: dict, ctx: str,
    ) -> str:
        return (
            "You are a helpful medical assistant answering questions for a patient about\n"
            "their prescriptions and health. The numbered references below are the ONLY\n"
            "things you may cite. Read each reference carefully.\n"
            "\n"
            "Citation rules (strict):\n"
            "1. After each factual claim, add a citation like [1] or [3].\n"
            "2. Cite a reference ONLY if its text directly states or supports the claim.\n"
            "   If no listed reference supports a claim, do not cite anything for it.\n"
            "3. Do not invent reference numbers. Do not cite [N] for a claim if [N]\n"
            "   talks about a different topic.\n"
            "4. You may cite multiple references for one claim ([1][3]).\n"
            "5. If none of the references answer the question, say so honestly.\n"
            "\n"
            f"{ctx}\n"
            "\n=== PATIENT QUESTION ===\n"
            f"{question}\n"
            "\nProvide a concise, helpful answer using only the references above for citations:"
        )

    def _verify_citations(self, answer: str, sources: list[dict],
                          threshold: float = -1.0) -> str:
        """
        For each [N] citation in the answer, score the surrounding sentence
        against the cited source via the cross-encoder. Drop citations whose
        score is below `threshold` (logit space).
        """
        if not self.encoders_ready or not sources:
            return answer
        sources_by_id = {s["id"]: s for s in sources}
        # Split answer into sentences (rough)
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        cleaned: list[str] = []
        for sent in sentences:
            cite_ids = [int(m) for m in re.findall(r"\[(\d+)\]", sent)]
            if not cite_ids:
                cleaned.append(sent)
                continue
            sent_text = re.sub(r"\[\d+\]", "", sent).strip()
            if not sent_text:
                cleaned.append(sent)
                continue
            kept_ids: list[int] = []
            for cid in cite_ids:
                src = sources_by_id.get(cid)
                if src is None:
                    continue
                pair = [[sent_text, src["text"][:512]]]
                enc = self.cross_tokenizer(
                    pair, truncation=True, padding=True,
                    max_length=512, return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    logits = self.cross_encoder(**enc).logits.squeeze()
                score = float(logits)
                if score >= threshold:
                    kept_ids.append(cid)
            # Replace original citations with kept ones
            cleaned_sent = re.sub(r"\[\d+\]", "", sent).rstrip()
            if kept_ids:
                cleaned_sent += " " + "".join(f"[{i}]" for i in sorted(set(kept_ids)))
            cleaned.append(cleaned_sent)
        return " ".join(cleaned)

    @staticmethod
    def _append_sources_block(answer: str, sources: list[dict]) -> str:
        """Append a 'Sources' footer to the LLM answer."""
        if not sources:
            return answer
        # Detect which sources the LLM actually cited
        cited_ids = set()
        for s in sources:
            if f"[{s['id']}]" in answer:
                cited_ids.add(s["id"])
        # If no citations detected, list all available sources
        listed = sorted(cited_ids) if cited_ids else [s["id"] for s in sources]
        lines = ["", "**Sources**"]
        for sid in listed:
            s = next((x for x in sources if x["id"] == sid), None)
            if s is None:
                continue
            if s["url"]:
                lines.append(f"[{sid}] {s['label']} — <{s['url']}>")
            else:
                lines.append(f"[{sid}] {s['label']}")
        return answer + "\n" + "\n".join(lines)

    def mode_3(self, question: str, patient: dict) -> dict:
        t0 = time.time()
        candidates = self.retrieve_patient_chunks(
            question, patient["patient_id"], top_k=4
        )
        reranked = self.rerank_chunks(question, candidates, top_k=3)
        ctx = self._format_patient_context(reranked)

        drug_names = [m["drug"] for m in patient.get("medications", [])]
        # Safety-prioritised KG triples (interactions and side effects first)
        drug_triples = self.kg_for_drugs(drug_names, top_k_per_drug=6,
                                          prioritize_safety=True)
        question_triples = self.kg_for_question(question, top_k=8)
        seen: set[tuple] = set()
        all_triples: list[dict] = []
        # Question-mentioned drug triples FIRST — they're directly about
        # what the user is asking, not just about the patient's prescriptions.
        for t in question_triples + drug_triples:
            key = (t["subject"], t["predicate"], t["object"])
            if key in seen:
                continue
            seen.add(key)
            all_triples.append(t)
        kg_block = self._format_kg_block(all_triples)

        # NEW: also pull MedlinePlus chunks for patient drugs + question drugs
        cand_m = self.retrieve_medlineplus_chunks(question, top_k=10, patient=patient)
        reranked_m_all = self.rerank_chunks(question, cand_m, top_k=10) if cand_m else []
        reranked_m = [
            c for c in reranked_m_all
            if c.get("cross_encoder_score", c.get("score", 0)) > -3.0
        ][:2]

        # Build numbered sources covering prescriptions, patient chunks,
        # MedlinePlus, and KG triples — every fact is independently citable.
        sources = self._build_sources_mode3(reranked, reranked_m, all_triples, patient)
        prompt = self._build_prompt_mode3_with_citations(question, patient, sources)
        answer = self.query_gemma(prompt)
        answer = self._verify_citations(answer, sources, threshold=-1.0)
        answer_with_refs = self._append_sources_block(answer, sources)

        return {
            "mode": 3, "mode_name": "RAG_KG",
            "answer": answer_with_refs,
            "latency_seconds": round(time.time() - t0, 2),
            "n_retrieved_chunks": len(reranked),
            "n_medlineplus_chunks": len(reranked_m),
            "n_kg_triples": len(all_triples),
            "retrieved_chunks": [
                {
                    "section": c["section"],
                    "score": c.get("cross_encoder_score", c["score"]),
                    "text": c["text"][:200],
                }
                for c in reranked
            ],
            "medlineplus_chunks": [
                {
                    "title": c["title"], "url": c.get("url", ""),
                    "score": c.get("cross_encoder_score", c["score"]),
                    "text": c["text"][:200],
                }
                for c in reranked_m
            ],
            "kg_triples": [t["triple_text"] for t in all_triples],
            "sources": [
                {"id": s["id"], "label": s["label"], "url": s.get("url", "")}
                for s in sources
            ],
        }
