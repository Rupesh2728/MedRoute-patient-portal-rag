"""Check what MedlinePlus retrieval returns for the ibuprofen+lisinopril question."""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "streamlit_app"))

from lib.pipeline import LiveBackend  # noqa: E402

backend = LiveBackend(PROJECT)
backend.load()

if not backend.medline_ready:
    print("MedlinePlus NOT loaded:", backend.warnings)
    sys.exit(1)
print(f"MedlinePlus loaded: {len(backend.medline_chunks)} chunks")

# Mock patient like Alice (P001) — metformin + lisinopril
patient = {
    "patient_id": "P001",
    "medications": [
        {"drug": "Metformin 1000mg"},
        {"drug": "Lisinopril 10mg"},
    ],
}
question = "Can I take ibuprofen with my lisinopril?"

# 1. Raw retrieval
print("\n" + "=" * 60)
print("retrieve_medlineplus_chunks(top_k=10, patient=patient)")
print("=" * 60)
cand_m = backend.retrieve_medlineplus_chunks(question, top_k=10, patient=patient)
print(f"Returned {len(cand_m)} candidates (after drug-name filter):")
for c in cand_m:
    print(f"  bi-score={c['score']:+.3f}  title='{c['title']}'  matched_terms={c.get('text', '')[:80]}…")

# 2. After cross-encoder rerank
print("\n" + "=" * 60)
print("After cross-encoder rerank (top_k=10)")
print("=" * 60)
reranked_m_all = backend.rerank_chunks(question, cand_m, top_k=10) if cand_m else []
for c in reranked_m_all:
    print(f"  cross-score={c['cross_encoder_score']:+.3f}  bi-score={c['score']:+.3f}  '{c['title']}'")

# 3. After threshold filter (-3.0)
print("\n" + "=" * 60)
print("After threshold filter (cross_encoder_score > -3.0, top 2)")
print("=" * 60)
reranked_m = [
    c for c in reranked_m_all
    if c.get("cross_encoder_score", c.get("score", 0)) > -3.0
][:2]
print(f"Final {len(reranked_m)} chunks would go to Mode 3 prompt:")
for c in reranked_m:
    print(f"  '{c['title']}'")
