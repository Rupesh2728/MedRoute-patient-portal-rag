"""
One-shot local setup:
1. Download MedCPT (Query + Article + Cross) to models/
2. Build patient FAISS for the 50 patients in patients.jsonl
3. Chunk + index 138 MedlinePlus articles
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

PROJECT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT / "models"
PATIENT_INDEX_DIR = PROJECT / "patient_index"
MEDLINE_INDEX_DIR = PROJECT / "medlineplus_index"
PATIENTS_PATH = PROJECT / "synthetic_patients" / "patients_v2.jsonl"  # 400 patients
MEDLINE_ARTICLES_PATH = PROJECT / "medlineplus_articles.jsonl"

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Device: {DEVICE}")


# ===========================================================================
# 1. Download MedCPT
# ===========================================================================

def ensure_medcpt() -> dict:
    """Download MedCPT models if not present."""
    MODELS_DIR.mkdir(exist_ok=True)
    out = {}
    for hf_name, local_name in [
        ("ncbi/MedCPT-Query-Encoder", "MedCPT-Query-Encoder"),
        ("ncbi/MedCPT-Article-Encoder", "MedCPT-Article-Encoder"),
        ("ncbi/MedCPT-Cross-Encoder", "MedCPT-Cross-Encoder"),
    ]:
        local_path = MODELS_DIR / local_name
        if local_path.exists() and any(local_path.iterdir()):
            print(f"  [skip] {local_name} already exists")
        else:
            print(f"  [download] {hf_name} -> {local_path}")
            tok = AutoTokenizer.from_pretrained(hf_name)
            tok.save_pretrained(local_path)
            if "Cross" in hf_name:
                model = AutoModelForSequenceClassification.from_pretrained(hf_name)
            else:
                model = AutoModel.from_pretrained(hf_name)
            model.save_pretrained(local_path)
        out[local_name] = local_path
    return out


# ===========================================================================
# Helpers
# ===========================================================================

def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def encode_texts(texts: list[str], tokenizer, model, batch_size: int = 16,
                 max_length: int = 256) -> np.ndarray:
    """Encode chunk texts -> normalized vectors."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True,
                        max_length=max_length, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
            emb = out.last_hidden_state[:, 0, :]
            emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu().numpy())
        if (i // batch_size) % 10 == 0:
            print(f"    encoded {min(i + batch_size, len(texts))}/{len(texts)}")
    return np.vstack(all_embs)


# ===========================================================================
# 2. Build patient FAISS
# ===========================================================================

def patient_to_chunks(patient: dict) -> list[dict]:
    chunks = []
    pid = patient["patient_id"]
    name = patient["name"]
    age = patient["age"]
    gender = patient["gender"].lower()

    # Demographics + active conditions
    conditions = patient.get("active_conditions", [])
    if conditions:
        cond_str = "; ".join(
            f"{c['condition']} (diagnosed {c['diagnosed']})" for c in conditions
        )
        text = (f"{name} is a {age}-year-old {gender} with the following active "
                f"medical conditions: {cond_str}.")
    else:
        text = f"{name} is a {age}-year-old {gender} with no active medical conditions documented."
    chunks.append({"chunk_id": f"{pid}_demographics_conditions", "patient_id": pid,
                   "section": "demographics_and_conditions", "text": text})

    # Past medical history
    pmh = patient.get("past_medical_history", [])
    if pmh:
        chunks.append({"chunk_id": f"{pid}_pmh", "patient_id": pid,
                       "section": "past_medical_history",
                       "text": f"{name} has the following past medical and surgical history: " + "; ".join(pmh) + "."})

    # Allergies
    allergies = patient.get("allergies", [])
    if allergies:
        text = f"{name} has the following documented drug or substance allergies: " + \
               "; ".join(f"{a['substance']} ({a.get('reaction', 'unspecified')})"
                          for a in allergies) + "."
    else:
        text = f"{name} has no documented drug allergies."
    chunks.append({"chunk_id": f"{pid}_allergies", "patient_id": pid,
                   "section": "allergies", "text": text})

    # Vitals
    vitals = patient.get("recent_vitals", {})
    if vitals:
        date = vitals.get("date", "the most recent visit")
        parts = [f"{k.replace('_', ' ')} {v}"
                 for k, v in vitals.items() if k != "date"]
        chunks.append({"chunk_id": f"{pid}_vitals", "patient_id": pid,
                       "section": "recent_vitals",
                       "text": f"{name}'s recent vitals and laboratory values from {date}: "
                               + ", ".join(parts) + "."})

    # Lifestyle
    lifestyle = patient.get("lifestyle", {})
    if lifestyle:
        parts = [f"{k.replace('_', ' ')}: {v}" for k, v in lifestyle.items()]
        chunks.append({"chunk_id": f"{pid}_lifestyle", "patient_id": pid,
                       "section": "lifestyle",
                       "text": f"{name}'s lifestyle factors. " + ". ".join(parts) + "."})
    return chunks


def build_patient_index(article_tok, article_model) -> None:
    PATIENT_INDEX_DIR.mkdir(exist_ok=True)
    patients = load_jsonl(PATIENTS_PATH)
    print(f"  Loaded {len(patients)} patients")

    chunks = []
    for p in patients:
        chunks.extend(patient_to_chunks(p))
    print(f"  Built {len(chunks)} chunks")

    texts = [c["text"] for c in chunks]
    embs = encode_texts(texts, article_tok, article_model)
    print(f"  Embeddings: {embs.shape}")

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype(np.float32))
    faiss.write_index(index, str(PATIENT_INDEX_DIR / "patient_index.bin"))
    with (PATIENT_INDEX_DIR / "patient_chunks.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    with (PATIENT_INDEX_DIR / "patient_chunk_ids.json").open("w") as f:
        json.dump([c["chunk_id"] for c in chunks], f)
    print(f"  Saved patient FAISS to {PATIENT_INDEX_DIR}")


# ===========================================================================
# 3. Build MedlinePlus FAISS
# ===========================================================================

def chunk_article(article: dict, target_words: int = 220) -> list[dict]:
    """Split an article into ~220-word paragraph-aligned chunks."""
    paragraphs = [p.strip() for p in article["text"].split("\n") if p.strip()]
    chunks = []
    cur_words: list[str] = []
    cur_paras: list[str] = []
    for p in paragraphs:
        pw = p.split()
        if len(cur_words) + len(pw) > target_words and cur_words:
            chunks.append("\n".join(cur_paras))
            cur_words, cur_paras = [], []
        cur_words.extend(pw)
        cur_paras.append(p)
    if cur_words:
        chunks.append("\n".join(cur_paras))

    out = []
    for i, txt in enumerate(chunks):
        out.append({
            "chunk_id": f"{article['article_id']}_c{i}",
            "article_id": article["article_id"],
            "title": article["title"],
            "url": article["url"],
            "topic_type": article.get("topic_type", "topic"),
            "matched_terms": article.get("matched_terms", []),
            "text": txt,
            "n_words": len(txt.split()),
        })
    return out


def build_medlineplus_index(article_tok, article_model) -> None:
    MEDLINE_INDEX_DIR.mkdir(exist_ok=True)
    articles = load_jsonl(MEDLINE_ARTICLES_PATH)
    print(f"  Loaded {len(articles)} MedlinePlus articles")

    all_chunks: list[dict] = []
    for art in articles:
        all_chunks.extend(chunk_article(art))
    print(f"  Chunked into {len(all_chunks)} passages "
          f"(median {int(np.median([c['n_words'] for c in all_chunks]))} words)")

    texts = [c["text"] for c in all_chunks]
    embs = encode_texts(texts, article_tok, article_model)
    print(f"  Embeddings: {embs.shape}")

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs.astype(np.float32))
    faiss.write_index(index, str(MEDLINE_INDEX_DIR / "medlineplus_index.bin"))
    with (MEDLINE_INDEX_DIR / "medlineplus_chunks.jsonl").open("w") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")
    print(f"  Saved MedlinePlus FAISS to {MEDLINE_INDEX_DIR}")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("=" * 60)
    print("STEP 1: Download MedCPT models")
    print("=" * 60)
    paths = ensure_medcpt()

    print("\n" + "=" * 60)
    print("STEP 2: Build patient FAISS (50 patients)")
    print("=" * 60)
    art_tok = AutoTokenizer.from_pretrained(str(paths["MedCPT-Article-Encoder"]))
    art_model = AutoModel.from_pretrained(str(paths["MedCPT-Article-Encoder"])).to(DEVICE).eval()
    build_patient_index(art_tok, art_model)

    print("\n" + "=" * 60)
    print("STEP 3: Build MedlinePlus FAISS (138 articles)")
    print("=" * 60)
    build_medlineplus_index(art_tok, art_model)

    print("\nAll done.")


if __name__ == "__main__":
    main()
