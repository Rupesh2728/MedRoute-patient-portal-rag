#!/usr/bin/env python3
"""Fetch MedlinePlus articles for drugs only"""

import json
import time
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, List

import requests
from bs4 import BeautifulSoup

API_BASE = 'https://wsearch.nlm.nih.gov/ws/query'
MEDLINEPLUS_BASE = 'https://medlineplus.gov'
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

OUTPUT_DIR = Path('/tmp/medlineplus')
OUTPUT_FILE = OUTPUT_DIR / 'medlineplus_articles.jsonl'
PROGRESS_FILE = OUTPUT_DIR / 'fetch_progress.json'

# Load existing progress
with open(PROGRESS_FILE) as f:
    progress = json.load(f)

fetched_urls = set(progress.get('fetched_urls', []))
article_counter = progress.get('article_counter', 0)
article_id_map = progress.get('article_id_map', {})

print(f"Starting from article counter: {article_counter}")
print(f"Already fetched: {len(fetched_urls)} URLs")


def fetch_html(url: str) -> str | None:
    """Fetch and extract text."""
    try:
        time.sleep(0.3)
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return None
        soup = BeautifulSoup(response.text, 'html.parser')
        for elem in soup(['script', 'style', 'nav', 'footer']):
            elem.decompose()
        text = soup.get_text(separator='\n', strip=True)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        return '\n'.join(lines)[:50000]
    except:
        return None


def query_api(term: str) -> List[Dict]:
    """Query API."""
    try:
        time.sleep(0.5)
        response = requests.get(API_BASE, params={
            'db': 'healthTopics',
            'term': term,
            'rettype': 'topic',
            'retmax': 5
        }, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return []

        root = ET.fromstring(response.text)
        results = []

        for doc in root.findall('.//document'):
            ht = doc.find('.//health-topic')
            if ht is not None:
                title = ht.get('title')
                url = ht.get('url')
                if title and url:
                    results.append({'title': title, 'url': url})

        return results
    except:
        return []


def save_article(article_data: Dict, matched_term: str) -> str | None:
    """Save article."""
    global article_counter, article_id_map

    url = article_data['url']

    if url in fetched_urls:
        return article_id_map.get(url)

    text = fetch_html(url)
    if not text:
        return None

    article_id = f"ml_{article_counter:05d}"
    article_counter += 1

    record = {
        'article_id': article_id,
        'title': article_data.get('title', 'Unknown'),
        'url': url,
        'source': 'MedlinePlus',
        'matched_terms': [matched_term],
        'topic_type': 'drug',
        'text': text,
        'n_words': len(text.split())
    }

    with open(OUTPUT_FILE, 'a') as f:
        f.write(json.dumps(record) + '\n')

    fetched_urls.add(url)
    article_id_map[url] = article_id

    print(f"[{article_id}] {record['title'][:50]:<50} ({record['n_words']:>5} words)")

    return article_id


# Extract drugs
drugs = set()
for fpath in [
    '/sessions/zealous-relaxed-keller/mnt/DL Final Project/synthetic_patients/patients.jsonl',
    '/sessions/zealous-relaxed-keller/mnt/DL Final Project/synthetic_patients/patients_v2.jsonl'
]:
    try:
        with open(fpath) as f:
            for line in f:
                patient = json.loads(line)
                for med in patient.get('medications', []):
                    drugs.add(med['drug'])
    except:
        pass

print(f"\nFetching {len(drugs)} unique drugs...")
print("=" * 80)

failed = []
for i, drug in enumerate(sorted(drugs), 1):
    if i % 20 == 0:
        print(f"[{i}/{len(drugs)}]", end='', flush=True)
        print()

    results = []

    # Try normalized queries
    variants = [drug.lower()]
    clean = re.sub(r'\s*\([^)]*\)\s*', ' ', drug).strip().lower()
    if clean != drug.lower():
        variants.append(clean)

    for variant in variants:
        api_results = query_api(variant)
        results.extend(api_results)
        if api_results:
            break

    if not results:
        failed.append(drug)
        continue

    for result in results:
        if result['url'] not in fetched_urls:
            save_article(result, drug)

# Save progress
progress = {
    'fetched_urls': list(fetched_urls),
    'article_counter': article_counter,
    'article_id_map': article_id_map
}
with open(PROGRESS_FILE, 'w') as f:
    json.dump(progress, f)

print("\n" + "=" * 80)
print(f"Done! Fetched {article_counter} total articles")
print(f"Drugs with no results: {len(failed)}")

# Summary
with open(OUTPUT_FILE) as f:
    articles = [json.loads(line) for line in f]

total_words = sum(a['n_words'] for a in articles)
print(f"\nFinal stats:")
print(f"  Total articles: {len(articles)}")
print(f"  Total words: {total_words:,}")
print(f"  Conditions: {len([a for a in articles if a['topic_type'] == 'condition'])}")
print(f"  Drugs: {len([a for a in articles if a['topic_type'] == 'drug'])}")
