"""
Drug-specific MedlinePlus fetcher (v2).

Strategy: scrape the canonical MedlinePlus alphabetical drug index pages,
build a (drug_name -> URL) lookup of every drug MedlinePlus catalogs (~3000),
then fetch the page for every drug in our patient cohort.

Adds new articles to medlineplus_articles.jsonl (deduplicated by URL).
After running this, re-run scripts/setup_local_pipeline.py to rebuild the
FAISS index over the bigger article set.
"""
from __future__ import annotations

import json
import re
import string
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

PROJECT = Path(__file__).resolve().parent.parent
PATIENTS_PATH = PROJECT / "synthetic_patients" / "patients_v2.jsonl"
ARTICLES_PATH = PROJECT / "medlineplus_articles.jsonl"

UA = {"User-Agent": "Mozilla/5.0 (research; medical Q&A project)"}
INDEX_URL = "https://medlineplus.gov/druginfo/drug_{letter}a.html"  # e.g. drug_Sa.html


# ---------------------------------------------------------------------------
# Patient drug names
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def unique_drug_names() -> list[str]:
    patients = load_jsonl(PATIENTS_PATH)
    drugs: set[str] = set()
    for p in patients:
        for m in p.get("medications", []):
            name = m["drug"].split()[0].strip().lower()
            name = re.sub(r"[^a-z]", "", name)
            if name:
                drugs.add(name)
    return sorted(drugs)


def existing_urls() -> set[str]:
    if not ARTICLES_PATH.exists():
        return set()
    arts = load_jsonl(ARTICLES_PATH)
    return {a["url"] for a in arts}


# ---------------------------------------------------------------------------
# Build drug name -> MedlinePlus URL map by scraping alphabetical index
# ---------------------------------------------------------------------------

def build_drug_url_map() -> dict[str, str]:
    """
    Scrape MedlinePlus alphabetical drug index. URL pattern:
      https://medlineplus.gov/druginfo/drug_<L>a.html for L in A..Z
    Each page lists drug names as <a href="meds/<id>.html">Name</a>.
    """
    url_map: dict[str, str] = {}
    for letter in string.ascii_uppercase:
        url = INDEX_URL.format(letter=letter)
        try:
            r = requests.get(url, headers=UA, timeout=15)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "meds/" in href and href.endswith(".html"):
                    name = a.get_text(strip=True).lower()
                    name = re.sub(r"[^a-z]", "", name)
                    if not name:
                        continue
                    full_url = (
                        href if href.startswith("http")
                        else f"https://medlineplus.gov/druginfo/{href.lstrip('/')}"
                    )
                    url_map.setdefault(name, full_url)
        except Exception as e:
            print(f"  index error for {letter}: {e}")
        time.sleep(0.3)
        print(f"  scanned letter {letter}: {len(url_map)} drugs in map so far")
    print(f"\nFinal drug-URL map: {len(url_map)} entries")
    return url_map


# ---------------------------------------------------------------------------
# Fetch drug article text
# ---------------------------------------------------------------------------

def fetch_html_text(url: str) -> tuple[str, str]:
    """Return (title, body_text) from a MedlinePlus page."""
    try:
        r = requests.get(url, timeout=15, headers=UA)
        r.raise_for_status()
    except Exception:
        return "", ""
    soup = BeautifulSoup(r.text, "html.parser")
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup
    text = main.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import sys
    fetch_all = "--all" in sys.argv

    cohort_drugs = unique_drug_names()
    print(f"Unique drug name stems in patient cohort: {len(cohort_drugs)}")

    existing = existing_urls()
    print(f"Existing articles in corpus: {len(existing)}")

    print("\nBuilding MedlinePlus drug-URL map...")
    url_map = build_drug_url_map()

    if ARTICLES_PATH.exists():
        existing_arts = load_jsonl(ARTICLES_PATH)
        max_id = 0
        for a in existing_arts:
            m = re.search(r"(\d+)", a["article_id"])
            if m:
                max_id = max(max_id, int(m.group(1)))
        next_id = max_id + 1
    else:
        next_id = 0

    if fetch_all:
        # Fetch every drug MedlinePlus has
        cohort_set = set(cohort_drugs)
        matched = sorted(
            url_map.items(),
            key=lambda kv: (0 if kv[0] in cohort_set else 1, kv[0]),
        )  # cohort drugs first, then everything else
        missed = []
        print(f"\nMode: ALL drugs ({len(matched)} candidates)")
    else:
        matched, missed = [], []
        for drug in cohort_drugs:
            if drug in url_map:
                matched.append((drug, url_map[drug]))
            else:
                missed.append(drug)
        print(f"\nMode: COHORT only ({len(matched)} of {len(cohort_drugs)} matched)")
    if missed:
        print(f"Missed (not in MedlinePlus): {missed[:10]}{'…' if len(missed) > 10 else ''}")

    new_articles = []
    for drug, url in matched:
        if url in existing or url in {a["url"] for a in new_articles}:
            print(f"  [dup] {drug}")
            continue
        title, text = fetch_html_text(url)
        if not text or len(text.split()) < 50:
            print(f"  [empty] {drug} {url}")
            continue
        new_articles.append({
            "article_id": f"ml_{next_id:05d}",
            "title": title or drug.title(),
            "url": url,
            "source": "MedlinePlus",
            "matched_terms": [drug],
            "topic_type": "drug",
            "text": text,
            "n_words": len(text.split()),
        })
        print(f"  [new] {drug} -> {title} ({len(text.split())} words)")
        next_id += 1
        time.sleep(0.4)

    print(f"\nFetched {len(new_articles)} new drug articles")
    if new_articles:
        with ARTICLES_PATH.open("a") as f:
            for a in new_articles:
                f.write(json.dumps(a) + "\n")
        print(f"Appended to {ARTICLES_PATH}")
    print(f"Total articles in file: {len(existing) + len(new_articles)}")


if __name__ == "__main__":
    main()
