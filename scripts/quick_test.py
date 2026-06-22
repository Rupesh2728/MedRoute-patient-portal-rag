#!/usr/bin/env python3
"""Quick test of the MedlinePlus fetcher on a small subset"""

import json
import time
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Set, List, Dict

import requests
from bs4 import BeautifulSoup

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
API_BASE = 'https://wsearch.nlm.nih.gov/ws/query'

# Test a few key terms
test_terms = {
    'conditions': ['Type 2 Diabetes Mellitus', 'Hypertension', 'Asthma'],
    'drugs': ['metformin', 'lisinopril', 'aspirin']
}

print("Testing MedlinePlus API and HTML fetching on subset...\n")

# Test API
print("=" * 60)
print("API Query Tests")
print("=" * 60)

for term in test_terms['conditions'][:1]:
    print(f"\nQuerying: {term}")
    try:
        time.sleep(0.5)
        response = requests.get(API_BASE, params={
            'db': 'healthTopics',
            'term': term,
            'rettype': 'topic',
            'retmax': 3
        }, headers=HEADERS, timeout=10)

        root = ET.fromstring(response.text)
        results = []

        for doc in root.findall('.//document'):
            ht = doc.find('.//health-topic')
            if ht is not None:
                title = ht.get('title')
                url = ht.get('url')
                if title and url:
                    results.append((title, url))

        print(f"  Found {len(results)} results:")
        for title, url in results[:2]:
            print(f"    - {title}")
            print(f"      {url}")

            # Try fetching the page
            print(f"    Fetching HTML...")
            time.sleep(0.3)
            try:
                hresp = requests.get(url, headers=HEADERS, timeout=10)
                if hresp.status_code == 200:
                    soup = BeautifulSoup(hresp.text, 'html.parser')
                    for script in soup(['script', 'style']):
                        script.decompose()
                    text = soup.get_text(separator='\n', strip=True)
                    lines = [l.strip() for l in text.split('\n') if l.strip()]
                    clean = '\n'.join(lines)[:500]
                    words = len(clean.split())
                    print(f"      OK - {words} words (sample): {clean[:100]}...")
                else:
                    print(f"      HTTP {hresp.status_code}")
            except Exception as e:
                print(f"      ERROR: {e}")

    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
