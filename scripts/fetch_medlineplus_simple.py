#!/usr/bin/env python3
"""
MedlinePlus Knowledge Base Fetcher - Simplified Version

Fast, sequential fetcher without threading complexity.
Uses API queries and direct drug page fetching.
"""

import json
import time
import re
import os
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, List
import xml.etree.ElementTree as ET
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

# Configuration
API_BASE = 'https://wsearch.nlm.nih.gov/ws/query'
MEDLINEPLUS_BASE = 'https://medlineplus.gov'
API_DELAY = 0.5  # seconds between API calls
HTML_DELAY = 0.3  # seconds between HTML fetches

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# Output directory - works in both regular and workspace environments
OUTPUT_DIR = Path('/tmp/medlineplus')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / 'medlineplus_articles.jsonl'
PROGRESS_FILE = OUTPUT_DIR / 'fetch_progress.json'

# Track progress
fetched_urls: Set[str] = set()
article_counter = 0
article_id_map: Dict[str, str] = {}


def load_progress():
    """Load previously fetched URLs."""
    global fetched_urls, article_counter, article_id_map
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE) as f:
                progress = json.load(f)
                fetched_urls = set(progress.get('fetched_urls', []))
                article_counter = progress.get('article_counter', 0)
                article_id_map = progress.get('article_id_map', {})
            print(f"[PROGRESS] Loaded: {len(fetched_urls)} previously fetched URLs, counter={article_counter}")
        except Exception as e:
            print(f"[WARN] Could not load progress: {e}")


def save_progress():
    """Save progress."""
    progress = {
        'fetched_urls': list(fetched_urls),
        'article_counter': article_counter,
        'article_id_map': article_id_map
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)


def extract_text_from_html(html: str) -> str:
    """Extract clean text from HTML."""
    soup = BeautifulSoup(html, 'html.parser')

    # Remove unwanted elements
    for elem in soup(['script', 'style', 'nav', 'footer', 'meta', 'link']):
        elem.decompose()

    # Get text
    text = soup.get_text(separator='\n', strip=True)

    # Clean excessive whitespace
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    clean_text = '\n'.join(lines)

    return clean_text[:50000]  # Cap at 50k


def fetch_html_page(url: str) -> str | None:
    """Fetch and extract text from HTML."""
    try:
        time.sleep(HTML_DELAY)
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            return None
        return extract_text_from_html(response.text)
    except Exception as e:
        print(f"[WARN] Error fetching {url}: {e}")
        return None


def query_api(term: str) -> List[Dict]:
    """Query MedlinePlus API."""
    try:
        time.sleep(API_DELAY)
        params = {
            'db': 'healthTopics',
            'term': term,
            'rettype': 'topic',
            'retmax': 5
        }
        response = requests.get(API_BASE, params=params, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return []

        root = ET.fromstring(response.text)
        results = []

        for doc in root.findall('.//document'):
            health_topic = doc.find('.//health-topic')
            if health_topic is not None:
                title = health_topic.get('title')
                url = health_topic.get('url')
                if title and url:
                    results.append({'title': title, 'url': url, 'source': 'api'})

        return results
    except Exception as e:
        print(f"[WARN] API error for '{term}': {e}")
        return []


def try_drug_page(drug_name: str) -> Dict | None:
    """Try direct drug info page."""
    normalized = re.sub(r'[^a-z0-9]', '_', drug_name.lower())
    normalized = re.sub(r'_+', '_', normalized).strip('_')
    url = f"{MEDLINEPLUS_BASE}/druginfo/meds/{normalized}.html"

    try:
        time.sleep(HTML_DELAY)
        response = requests.head(url, headers=HEADERS, timeout=5, allow_redirects=True)
        if response.status_code == 200:
            return {
                'title': f"{drug_name} - MedlinePlus",
                'url': response.url,
                'source': 'direct'
            }
    except:
        pass
    return None


def save_article(article_data: Dict, matched_term: str, term_type: str) -> str | None:
    """Save article to JSONL."""
    global article_counter, article_id_map

    url = article_data['url']

    if url in fetched_urls:
        return article_id_map.get(url)

    # Fetch text
    text = fetch_html_page(url)
    if not text:
        print(f"[SKIP] No text for {url}")
        return None

    # Generate ID
    article_id = f"ml_{article_counter:05d}"
    article_counter += 1

    # Build record
    record = {
        'article_id': article_id,
        'title': article_data.get('title', 'Unknown'),
        'url': url,
        'source': 'MedlinePlus',
        'matched_terms': [matched_term],
        'topic_type': term_type,
        'text': text,
        'n_words': len(text.split())
    }

    # Write to JSONL
    with open(OUTPUT_FILE, 'a') as f:
        f.write(json.dumps(record) + '\n')

    # Track
    fetched_urls.add(url)
    article_id_map[url] = article_id

    print(f"[{article_id}] {record['title'][:50]:<50} ({record['n_words']:>5} words)")

    return article_id


def extract_terms():
    """Extract unique terms from patient files."""
    conditions = set()
    drugs = set()

    files = [
        '/sessions/zealous-relaxed-keller/mnt/DL Final Project/synthetic_patients/patients.jsonl',
        '/sessions/zealous-relaxed-keller/mnt/DL Final Project/synthetic_patients/patients_v2.jsonl'
    ]

    for fpath in files:
        try:
            with open(fpath) as f:
                for line in f:
                    patient = json.loads(line)
                    for cond_obj in patient.get('active_conditions', []):
                        conditions.add(cond_obj['condition'])
                    for pmh in patient.get('past_medical_history', []):
                        conditions.add(pmh)
                    for med in patient.get('medications', []):
                        drugs.add(med['drug'])
            print(f"[OK] Loaded {Path(fpath).name}")
        except FileNotFoundError:
            print(f"[WARN] Not found: {fpath}")

    return conditions, drugs


def normalize_term(term: str) -> List[str]:
    """Generate search variants."""
    variants = [term.lower()]

    # Remove parenthetical
    clean = re.sub(r'\s*\([^)]*\)\s*', ' ', term).strip().lower()
    if clean != term.lower():
        variants.append(clean)

    # Handle specific patterns
    if ' - ' in term:
        variants.append(term.split(' - ')[0].lower())

    return list(dict.fromkeys(variants))


def main():
    """Main orchestration."""
    print("=" * 80)
    print("MedlinePlus Knowledge Base Fetcher (Simplified)")
    print("=" * 80)

    load_progress()

    # Extract terms
    print("\n[EXTRACT] Extracting terms from patient files...")
    conditions, drugs = extract_terms()

    print(f"\nFound {len(conditions)} unique conditions and {len(drugs)} unique drugs")
    print(f"Total terms to fetch: {len(conditions) + len(drugs)}")

    failed_terms = []
    term_article_count = defaultdict(int)

    # Fetch conditions
    print("\n" + "=" * 80)
    print(f"[FETCH] Querying {len(conditions)} conditions...")
    print("=" * 80)

    for i, term in enumerate(sorted(conditions), 1):
        if i % 20 == 0:
            print(f"[PROGRESS] {i}/{len(conditions)}")
            save_progress()

        results = []
        for variant in normalize_term(term):
            results = query_api(variant)
            if results:
                break

        if not results:
            failed_terms.append((term, 'condition'))
            continue

        for result in results:
            if result['url'] not in fetched_urls:
                article_id = save_article(result, term, 'condition')
                if article_id:
                    term_article_count[term] += 1

    # Fetch drugs
    print("\n" + "=" * 80)
    print(f"[FETCH] Querying {len(drugs)} drugs...")
    print("=" * 80)

    for i, drug in enumerate(sorted(drugs), 1):
        if i % 20 == 0:
            print(f"[PROGRESS] {i}/{len(drugs)}")
            save_progress()

        results = []

        # Try direct page first
        direct = try_drug_page(drug)
        if direct and direct['url'] not in fetched_urls:
            results.append(direct)

        # Try API
        if not results:
            for variant in normalize_term(drug):
                api_results = query_api(variant)
                results.extend(api_results)
                if api_results:
                    break

        if not results:
            failed_terms.append((drug, 'drug'))
            continue

        for result in results:
            if result['url'] not in fetched_urls:
                article_id = save_article(result, drug, 'drug')
                if article_id:
                    term_article_count[drug] += 1

    # Final save
    save_progress()

    # Report
    print("\n" + "=" * 80)
    print("FETCH COMPLETE")
    print("=" * 80)

    print(f"\nArticles fetched: {article_counter}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Output path for RAG: {OUTPUT_DIR}")

    # Calculate stats
    total_words = 0
    sample_articles = []
    try:
        with open(OUTPUT_FILE) as f:
            for i, line in enumerate(f):
                article = json.loads(line)
                total_words += article['n_words']
                if i < 3:
                    sample_articles.append((article['article_id'], article['title'], article['url']))
    except FileNotFoundError:
        pass

    print(f"Total words: {total_words:,}")

    if sample_articles:
        print("\nSample articles (first 3):")
        for aid, title, url in sample_articles:
            print(f"  [{aid}] {title}")
            print(f"       {url}")

    if failed_terms:
        print(f"\nTerms with no MedlinePlus results ({len(failed_terms)}):")
        for term, term_type in sorted(failed_terms)[:20]:
            print(f"  - {term_type}: {term}")
        if len(failed_terms) > 20:
            print(f"  ... and {len(failed_terms) - 20} more")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
