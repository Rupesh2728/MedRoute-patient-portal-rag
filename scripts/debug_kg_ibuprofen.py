"""Quick debug: check if ibuprofen exists in PrimeKG and what kg_for_question returns."""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "streamlit_app"))

from lib.pipeline import LiveBackend  # noqa: E402

backend = LiveBackend(PROJECT)
backend.load()

if not backend.kg_ready:
    print("KG not loaded:", backend.warnings)
    sys.exit(1)

# 1. Is ibuprofen in PrimeKG at all?
print("=" * 60)
print("PrimeKG entity check")
print("=" * 60)
for drug in ["ibuprofen", "Ibuprofen", "IBUPROFEN", "advil"]:
    in_kg = drug.lower() in backend.name_to_triples
    print(f"  '{drug}' in name_to_triples: {in_kg}")

# 2. Show first 10 ibuprofen triples (raw, unsorted)
if "ibuprofen" in backend.name_to_triples:
    print(f"\nIbuprofen has {len(backend.name_to_triples['ibuprofen'])} triples in KG")
    print("First 10 raw triples (no priority sorting):")
    for rel, y, t in backend.name_to_triples["ibuprofen"][:10]:
        print(f"  ({rel}) -> {y}  [type={t}]")

# 3. What does kg_for_question return for the ibuprofen+lisinopril question?
print("\n" + "=" * 60)
print("kg_for_question output")
print("=" * 60)
qts = backend.kg_for_question(
    "Can I take ibuprofen with my lisinopril?", top_k=8
)
print(f"\nReturned {len(qts)} triples:")
for t in qts:
    print(f"  {t['triple_text']}")

# 4. What does scispaCy extract from that question?
if backend.nlp is not None:
    print("\n" + "=" * 60)
    print("scispaCy entities in question")
    print("=" * 60)
    doc = backend.nlp("Can I take ibuprofen with my lisinopril?")
    for ent in doc.ents:
        print(f"  '{ent.text}' (label={ent.label_})")
