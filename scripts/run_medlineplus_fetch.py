#!/usr/bin/env python3
"""
Quick runner for MedlinePlus fetcher with better error handling and output
"""
import sys
import subprocess

result = subprocess.run([
    sys.executable,
    '/Users/maqsood/Documents/Claude/Projects/DL Final Project/scripts/fetch_medlineplus.py'
], timeout=3600)

sys.exit(result.returncode)
