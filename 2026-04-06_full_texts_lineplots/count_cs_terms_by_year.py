#!/usr/bin/env python3
"""
Count occurrences of ~25 CS-related terms per year across the JSTOR full-text corpus.
Also tracks total word tokens per year for computing relative frequencies.

Outputs: cs_terms_by_year.csv with columns:
    year, total_articles, total_tokens, programming, AI, ..., Pattern recognition

Usage:
    python count_cs_terms_by_year.py <input.jsonl> <metadata.csv>

Example:
    python count_cs_terms_by_year.py \
        e119d16a-1426-497d-a42a-a239d354c137.jsonl \
        ../jstor_analysis_code/jstor_humanities_extended_plus_ling_above_2000_full_metadata.csv
"""
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict

# Each entry: (display_label, regex_pattern_string)
CS_TERMS = [
    ("programming",                        r'\bprogramming\b'),
    ("AI",                                 r'\bAI\b'),
    ("Artificial intelligence",            r'\bartificial intelligence\b'),
    ("Character recognition",              r'\bcharacter recognition\b'),
    ("ChatGPT",                            r'\bchatgpt\b'),
    ("Clustering analysis",                r'\bclustering analysis\b'),
    ("Comput*",                            r'\bcomput.+?\b'),
    ("Computational",                      r'\bcomputational\b'),
    ("Digital",                            r'\bdigital\b'),
    ("Distant reading",                    r'\bdistant reading\b'),
    ("Entity Recognition",                 r'\bentity recognition\b'),
    ("GenAI",                              r'\bgenai\b'),
    ("Generative AI",                      r'\bgenerative ai\b'),
    ("Generative Artificial intelligence", r'\bgenerative artificial intelligence\b'),
    ("Humanities Computing",               r'\bhumanities computing\b'),
    ("Large language model",               r'\blarge language model\b'),
    ("LLM",                                r'\bLLM\b'),
    ("Literary Computing",                 r'\bliterary computing\b'),
    ("Machine learning",                   r'\bmachine learning\b'),
    ("Natural language processing",        r'\bnatural language processing\b'),
    ("NLP",                                r'\bNLP\b'),
    ("Named Entity Recognition",           r'\bnamed entity recognition\b'),
    ("Network analysis",                   r'\bnetwork analysis\b'),
    ("OCR",                                r'\bOCR\b'),
    ("Pattern recognition",                r'\bpattern recognition\b'),
]

TERM_LABELS = [label for label, _ in CS_TERMS]

# Simple word-token pattern for counting total tokens
WORD_RE = re.compile(r'\b[a-zA-Z]+\b')


def load_year_lookup(metadata_path):
    """Load metadata CSV into a dict: item_id -> year (int)."""
    lookup = {}
    with open(metadata_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lookup[row['item_id']] = int(row['year'])
            except (ValueError, KeyError):
                continue
    return lookup


def main(jsonl_path, metadata_path):
    # Step 1: Load year lookup from metadata
    print("Loading metadata...", file=sys.stderr)
    year_of = load_year_lookup(metadata_path)
    print(f"  Loaded {len(year_of):,} item-year mappings.", file=sys.stderr)

    # Step 2: Compile regex patterns
    case_sensitive_terms = {"AI", "LLM", "NLP", "OCR"}
    compiled = []
    for label, pat in CS_TERMS:
        flags = 0 if label in case_sensitive_terms else re.IGNORECASE
        compiled.append((label, re.compile(pat, flags)))

    # Step 3: Per-year accumulators
    articles_per_year = defaultdict(int)
    tokens_per_year = defaultdict(int)
    # term_counts_per_year[year][label] = total occurrences
    term_counts_per_year = defaultdict(lambda: defaultdict(int))

    no_year_count = 0
    total_items = 0
    start = time.time()

    # Step 4: Stream through JSONL
    with open(jsonl_path, encoding='utf-8') as f:
        for line in f:
            total_items += 1
            if total_items % 100_000 == 0:
                elapsed = time.time() - start
                print(f"  {total_items:,} processed ({elapsed:.0f}s)...",
                      file=sys.stderr)

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            iid = record.get('iid', '')
            year = year_of.get(iid)
            if year is None:
                no_year_count += 1
                continue

            combined = ' '.join(record.get('full_text', []))

            # Count total word tokens
            token_count = len(WORD_RE.findall(combined))
            articles_per_year[year] += 1
            tokens_per_year[year] += token_count

            # Count each CS term
            for label, pat in compiled:
                matches = pat.findall(combined)
                if matches:
                    term_counts_per_year[year][label] += len(matches)

    elapsed = time.time() - start

    # Step 5: Write output CSV
    output_path = os.path.join(os.path.dirname(jsonl_path), 'cs_terms_by_year.csv')
    years = sorted(articles_per_year.keys())

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        header = ['year', 'total_articles', 'total_tokens'] + TERM_LABELS
        writer.writerow(header)
        for y in years:
            row = [y, articles_per_year[y], tokens_per_year[y]]
            for label in TERM_LABELS:
                row.append(term_counts_per_year[y].get(label, 0))
            writer.writerow(row)

    # Print summary
    print(f"\nDone in {elapsed:.1f}s", file=sys.stderr)
    print(f"Total items processed: {total_items:,}", file=sys.stderr)
    print(f"Items matched to a year: {sum(articles_per_year.values()):,}", file=sys.stderr)
    print(f"Items with no year found: {no_year_count:,}", file=sys.stderr)
    print(f"Year range: {years[0]}--{years[-1]}", file=sys.stderr)
    print(f"Results saved to: {output_path}", file=sys.stderr)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.jsonl> <metadata.csv>",
              file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
