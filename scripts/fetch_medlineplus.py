#!/usr/bin/env python3
"""
MedlinePlus Knowledge Base Fetcher

Extracts unique conditions and drugs from patient cohort, fetches corresponding
MedlinePlus articles via Web Service API and direct HTML pages, deduplicates,
and saves structured JSON corpus for RAG.
"""

import json
import time
import logging
import re
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict, List, Tuple
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

# Use temp directory for logs if we can't write to main path
try:
    log_dir = Path('/Users/maqsood/Documents/Claude/Projects/DL Final Project/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'medlineplus_fetch.log'
except (PermissionError, FileNotFoundError):
    log_dir = Path('/tmp')
    log_file = log_dir / 'medlineplus_fetch.log'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(str(log_file)),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Output directory - use temp if primary not accessible
try:
    OUTPUT_DIR = Path('/Users/maqsood/Documents/Claude/Projects/DL Final Project/medlineplus')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
except (PermissionError, FileNotFoundError):
    OUTPUT_DIR = Path('/tmp/medlineplus')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / 'medlineplus_articles.jsonl'
PROGRESS_FILE = OUTPUT_DIR / 'fetch_progress.json'

# API constants
API_BASE = 'https://wsearch.nlm.nih.gov/ws/query'
MEDLINEPLUS_DRUG_BASE = 'https://medlineplus.gov/druginfo/meds'
MEDLINEPLUS_BASE = 'https://medlineplus.gov'

# Polite delays
API_DELAY = 0.5  # seconds between API calls
HTML_DELAY = 0.3  # seconds between HTML fetches

# Request headers
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# Track fetched URLs to avoid duplicates
fetched_urls: Set[str] = set()
article_counter = 0
article_id_map: Dict[str, str] = {}  # URL -> article_id


def load_progress():
    """Load previously fetched URLs and article counter from progress file."""
    global fetched_urls, article_counter, article_id_map
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE) as f:
                progress = json.load(f)
                fetched_urls = set(progress.get('fetched_urls', []))
                article_counter = progress.get('article_counter', 0)
                article_id_map = progress.get('article_id_map', {})
            logger.info(f"Loaded progress: {len(fetched_urls)} previously fetched URLs, counter at {article_counter}")
        except Exception as e:
            logger.warning(f"Could not load progress file: {e}")
    else:
        fetched_urls = set()
        article_counter = 0
        article_id_map = {}


def save_progress():
    """Save current progress to file."""
    progress = {
        'fetched_urls': list(fetched_urls),
        'article_counter': article_counter,
        'article_id_map': article_id_map
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


def extract_text_from_html(html: str) -> str:
    """Extract clean text from MedlinePlus HTML page."""
    soup = BeautifulSoup(html, 'html.parser')

    # Remove script and style elements
    for script in soup(['script', 'style', 'nav', 'footer']):
        script.decompose()

    # Try to find main content area
    main_content = None
    for container in ['mpl-document', 'content', 'main-content', 'page-content']:
        main_content = soup.find(class_=container)
        if main_content:
            break

    if not main_content:
        main_content = soup.body or soup

    # Get text
    text = main_content.get_text(separator='\n', strip=True)

    # Clean up excessive whitespace
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    clean_text = '\n'.join(lines)

    return clean_text[:50000]  # Cap at 50k chars per article


def fetch_html_page(url: str) -> str | None:
    """Fetch and extract text from an HTML page."""
    try:
        time.sleep(HTML_DELAY)
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 404:
            logger.warning(f"404 for {url}")
            return None
        if response.status_code != 200:
            logger.warning(f"HTTP {response.status_code} for {url}")
            return None
        return extract_text_from_html(response.text)
    except Exception as e:
        logger.warning(f"Error fetching {url}: {e}")
        return None


def query_medlineplus_api(term: str) -> List[Dict]:
    """Query MedlinePlus Web Service API for a term."""
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
            logger.warning(f"API returned {response.status_code} for term '{term}'")
            return []

        # Parse XML response with ElementTree
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as e:
            logger.warning(f"XML parse error for term '{term}': {e}")
            return []

        results = []

        # Navigate through the XML hierarchy: nlmSearchResult > list > document > content > health-topic
        for doc in root.findall('.//document'):
            health_topic = doc.find('.//health-topic')
            if health_topic is not None:
                title = health_topic.get('title')
                url = health_topic.get('url')
                if title and url:
                    results.append({
                        'title': title,
                        'url': url,
                        'source': 'api'
                    })

        return results
    except Exception as e:
        logger.warning(f"API error for term '{term}': {e}")
        return []


def try_drug_page(drug_name: str) -> Dict | None:
    """Try to fetch direct MedlinePlus drug info page."""
    # Normalize drug name: lowercase, replace spaces/dashes with underscores, remove special chars
    normalized = re.sub(r'[^a-z0-9]', '_', drug_name.lower())
    normalized = re.sub(r'_+', '_', normalized).strip('_')

    url = f"{MEDLINEPLUS_DRUG_BASE}/{normalized}.html"

    try:
        time.sleep(HTML_DELAY)
        response = requests.head(url, headers=HEADERS, timeout=5, allow_redirects=True)
        if response.status_code == 200:
            return {
                'title': f"{drug_name} - MedlinePlus",
                'url': response.url,  # Use final redirected URL
                'source': 'direct_page'
            }
    except Exception as e:
        pass  # Silently fail for missing pages

    return None


def normalize_search_term(term: str) -> List[str]:
    """Generate normalized versions of a search term for fallback searches."""
    variants = [term.lower()]

    # Remove common suffixes/patterns
    if ' - ' in term:
        variants.append(term.split(' - ')[0].lower())

    if ' (nondiabetic)' in term.lower():
        variants.append(term.lower().replace(' (nondiabetic)', ''))

    if ' (non-dysplastic)' in term.lower():
        variants.append(term.lower().replace(' (non-dysplastic)', ''))

    if ' (paroxysmal)' in term.lower():
        variants.append('atrial fibrillation')

    if ' (persistent)' in term.lower():
        variants.append('atrial fibrillation')

    # Remove parenthetical sections
    clean = re.sub(r'\s*\([^)]*\)\s*', ' ', term).strip().lower()
    if clean and clean != term.lower():
        variants.append(clean)

    # Remove leading numbers and surgical history markers
    if 'ectomy' in term.lower():
        base = re.sub(r'\s*\(\d+\)\s*', '', term).lower()
        if base != term.lower():
            variants.append(base)

    return list(dict.fromkeys(variants))  # Remove duplicates while preserving order


def fetch_for_term(term: str, is_drug: bool = False) -> List[Dict]:
    """Fetch articles for a single term (condition or drug)."""
    results = []

    logger.info(f"Fetching {'drug' if is_drug else 'condition'}: {term}")

    # For drugs, try direct page first
    if is_drug:
        direct = try_drug_page(term)
        if direct and direct['url'] not in fetched_urls:
            results.append(direct)

    # Try API queries with normalized variants
    search_variants = normalize_search_term(term)
    for variant in search_variants:
        if not variant.strip():
            continue

        api_results = query_medlineplus_api(variant)
        for result in api_results:
            if result['url'] not in fetched_urls:
                results.append(result)

        if results:  # Stop after first successful variant
            break

    return results


def save_article(article_data: Dict) -> str | None:
    """Save a single article and return its article_id."""
    global article_counter, article_id_map

    url = article_data['url']

    if url in fetched_urls:
        return article_id_map.get(url)

    # Fetch full HTML text
    text = fetch_html_page(url)
    if not text:
        logger.warning(f"Could not fetch text for {url}")
        return None

    # Generate article ID
    article_id = f"ml_{article_counter:05d}"
    article_counter += 1

    # Build output record
    record = {
        'article_id': article_id,
        'title': article_data.get('title', 'Unknown'),
        'url': url,
        'source': 'MedlinePlus',
        'matched_terms': article_data.get('matched_terms', []),
        'topic_type': article_data.get('topic_type', 'topic'),
        'text': text,
        'n_words': len(text.split())
    }

    # Append to JSONL file
    with open(OUTPUT_FILE, 'a') as f:
        f.write(json.dumps(record) + '\n')

    # Track as fetched
    fetched_urls.add(url)
    article_id_map[url] = article_id

    logger.info(f"[{article_id}] {article_data.get('title', 'Unknown')[:60]}... ({record['n_words']} words)")

    return article_id


def extract_terms_from_patients() -> Tuple[Set[str], Set[str]]:
    """Extract unique conditions and drugs from both patient files."""
    conditions = set()
    drugs = set()

    # Use mounted paths for workspace bash environment
    patient_files = [
        '/sessions/zealous-relaxed-keller/mnt/DL Final Project/synthetic_patients/patients.jsonl',
        '/sessions/zealous-relaxed-keller/mnt/DL Final Project/synthetic_patients/patients_v2.jsonl'
    ]

    for patient_file in patient_files:
        try:
            logger.info(f"Extracting terms from {Path(patient_file).name}...")
            with open(patient_file) as f:
                for i, line in enumerate(f):
                    patient = json.loads(line)

                    # Active conditions
                    for cond_obj in patient.get('active_conditions', []):
                        conditions.add(cond_obj['condition'])

                    # Past medical history
                    for pmh in patient.get('past_medical_history', []):
                        conditions.add(pmh)

                    # Medications
                    for med in patient.get('medications', []):
                        drugs.add(med['drug'])

            logger.info(f"  Found {len(conditions)} unique conditions, {len(drugs)} unique drugs so far")
        except FileNotFoundError:
            logger.warning(f"File not found: {patient_file}")

    return conditions, drugs


def main():
    """Main orchestration."""
    logger.info("=" * 80)
    logger.info("MedlinePlus Knowledge Base Fetcher")
    logger.info("=" * 80)

    # Load progress if resuming
    load_progress()

    # Extract terms from patients
    conditions, drugs = extract_terms_from_patients()

    logger.info(f"\nTotal unique conditions: {len(conditions)}")
    logger.info(f"Total unique drugs: {len(drugs)}")
    logger.info(f"Total terms to fetch: {len(conditions) + len(drugs)}")

    # Build fetch queue with term metadata
    fetch_queue = []
    for cond in sorted(conditions):
        fetch_queue.append((cond, False, 'condition'))
    for drug in sorted(drugs):
        fetch_queue.append((drug, True, 'drug'))

    # Track results
    failed_terms = []
    article_count_by_term = defaultdict(int)

    logger.info(f"\nStarting fetch with up to 5 concurrent workers...")

    # Use ThreadPoolExecutor for concurrent fetches
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}

        for term, is_drug, term_type in fetch_queue:
            future = executor.submit(fetch_for_term, term, is_drug)
            futures[future] = (term, is_drug, term_type)

        for i, future in enumerate(as_completed(futures), 1):
            term, is_drug, term_type = futures[future]

            try:
                results = future.result()

                if not results:
                    logger.warning(f"[{i}/{len(fetch_queue)}] No results for {term_type} '{term}'")
                    failed_terms.append((term, term_type))
                    continue

                # Save each unique result
                for result in results:
                    result['matched_terms'] = [term]
                    result['topic_type'] = term_type
                    article_id = save_article(result)
                    if article_id:
                        article_count_by_term[term] += 1

                        if (i % 20 == 0):
                            save_progress()

            except Exception as e:
                logger.error(f"Error processing {term}: {e}")
                failed_terms.append((term, term_type))

    # Final save
    save_progress()

    # Report
    logger.info("\n" + "=" * 80)
    logger.info("FETCH COMPLETE")
    logger.info("=" * 80)

    logger.info(f"Total articles fetched: {article_counter}")
    logger.info(f"Output file: {OUTPUT_FILE}")
    logger.info(f"Total unique URLs fetched: {len(fetched_urls)}")

    # Calculate total word count
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

    logger.info(f"Total word count: {total_words:,}")

    if sample_articles:
        logger.info("\nSample articles (first 3):")
        for aid, title, url in sample_articles:
            logger.info(f"  [{aid}] {title}")
            logger.info(f"       {url}")

    if failed_terms:
        logger.info(f"\nTerms with no MedlinePlus results ({len(failed_terms)}):")
        for term, term_type in sorted(failed_terms)[:20]:
            logger.info(f"  - {term_type}: {term}")
        if len(failed_terms) > 20:
            logger.info(f"  ... and {len(failed_terms) - 20} more")

    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    main()
