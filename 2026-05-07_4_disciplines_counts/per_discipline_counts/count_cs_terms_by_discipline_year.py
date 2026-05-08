#!/usr/bin/env python3
"""
Per-discipline, per-year CS-term counts for single-discipline papers in
4 target disciplines: Language & Literature, Linguistics, History, Archaeology.

Three count modes are produced for every (discipline, year, term):
  1. raw_min3            -- sum of in-paper occurrences, but only for papers
                            where the term appears >= 3 times. Papers with
                            1 or 2 occurrences contribute 0.
  2. once_per_paper      -- 1 per paper containing the term at least once.
  3. once_per_paper_min3 -- 1 per paper where the term appears >= 3 times.

Reference sections are stripped before counting, using the same heuristic as
`../no_references_experiment/count_cs_terms_by_year_no_refs.py`.

Output layout (relative to this script):
    raw_min3/<discipline>.csv
    once_per_paper/<discipline>.csv
    once_per_paper_min3/<discipline>.csv
    run_stats.txt

Each CSV has columns: year, total_articles, total_tokens, <25 term columns>.
total_articles / total_tokens are identical across the three modes for a
given discipline-year (they reflect the post-strip exposure).

Usage:
    python count_cs_terms_by_discipline_year.py <input.jsonl> <metadata.csv>

Example:
    python count_cs_terms_by_discipline_year.py \
        ../e119d16a-1426-497d-a42a-a239d354c137.jsonl \
        ../../jstor_analysis_code/jstor_humanities_extended_plus_ling_above_2000_full_metadata.csv
"""
import ast
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict

TARGET_DISCIPLINES = [
    "Language & Literature",
    "Linguistics",
    "History",
    "Archaeology",
]

DISCIPLINE_SLUGS = {
    "Language & Literature": "language_and_literature",
    "Linguistics":           "linguistics",
    "History":               "history",
    "Archaeology":           "archaeology",
}

DISCIPLINE_SHORT = {
    "Language & Literature": "L&L",
    "Linguistics":           "Ling",
    "History":               "Hist",
    "Archaeology":           "Archaeo",
}

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

WORD_RE = re.compile(r'\b[a-zA-Z]+\b')

# Reference-section heuristic, redesigned for the joined-paragraph form of
# this corpus. The JSONL stores `full_text` as a list of paragraphs; we join
# them with a single space, which strips paragraph newlines and double-spaces.
# The original heuristic in count_cs_terms_by_year_no_refs.py required `\n`
# or `\s{2,}` around the marker, which never matches after the join — so
# nothing was being stripped (verified empirically on a 5k slice: 0%).
#
# Replacement strategy: look for one of the standard heading words/phrases
# IMMEDIATELY followed by a citation-start pattern. The citation pattern
# is what actually distinguishes a section heading from an in-body mention:
#   - "Smith,"  /  "Acosta-Belén,"  /  "Bakshi."   — Capital-word + comma/period
#   - "1." / "12 ."                                — numbered reference
#   - "A. Smith"                                   — initial-and-name
# Body uses ("the references to...", "bibliography of his works",
# "References Smith (1990) found...") fail the citation lookahead because
# they're followed by lowercase function words or by parens / non-comma chars.
# The "last 40%" cutoff is kept as additional insurance against early FPs.
_MARKER = (
    r'(?P<marker>(?i:notes\s+and\s+references|select(?:ed)?\s+bibliography'
    r'|references\s+cited|literature\s+cited|works\s+cited'
    r'|references|bibliography|bibliographie))'
)
_CITE = r"(?:[A-Z][\w\-'À-ſ]+[,.]|\d+\s*\.|[A-Z]\.\s+[A-Z])"
REFERENCE_RE = re.compile(rf'\b{_MARKER}\s+{_CITE}')
REFERENCE_CUTOFF = 0.6  # marker must appear in last 40% of text


def strip_references(text):
    """Remove the references section from the end of an article.
    Returns (stripped_text, was_stripped)."""
    cutoff_pos = int(len(text) * REFERENCE_CUTOFF)
    tail = text[cutoff_pos:]
    match = REFERENCE_RE.search(tail)
    if match:
        return text[:cutoff_pos + match.start('marker')], True
    return text, False


def load_metadata(metadata_path):
    """Build iid -> (year, target_discipline_or_None).

    target_discipline is set only when the paper's discipline list contains
    exactly one entry AND that entry is in TARGET_DISCIPLINES. Other rows are
    skipped on the JSONL pass.
    """
    targets = set(TARGET_DISCIPLINES)
    lookup = {}
    per_disc = defaultdict(int)
    total = 0
    multi = 0
    no_year = 0

    with open(metadata_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            try:
                year = int(row['year'])
            except (ValueError, KeyError, TypeError):
                no_year += 1
                continue
            try:
                discs = ast.literal_eval(row['discipline_names'])
            except (ValueError, SyntaxError):
                continue
            unique = sorted(set(discs))
            if len(unique) == 1 and unique[0] in targets:
                disc = unique[0]
                lookup[row['item_id']] = (year, disc)
                per_disc[disc] += 1
            elif len(unique) > 1:
                multi += 1

    return lookup, dict(per_disc), {
        'total_metadata_rows': total,
        'no_year': no_year,
        'multi_discipline': multi,
    }


def main(jsonl_path, metadata_path):
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("Loading metadata...", file=sys.stderr)
    meta_lookup, per_disc_pre, meta_stats = load_metadata(metadata_path)
    print(f"  Metadata rows: {meta_stats['total_metadata_rows']:,}", file=sys.stderr)
    print(f"  Single-discipline target items: {len(meta_lookup):,}", file=sys.stderr)
    for disc in TARGET_DISCIPLINES:
        print(f"    {disc:<25} {per_disc_pre.get(disc, 0):>8,}", file=sys.stderr)

    case_sensitive_terms = {"AI", "LLM", "NLP", "OCR"}
    compiled = []
    for label, pat in CS_TERMS:
        flags = 0 if label in case_sensitive_terms else re.IGNORECASE
        compiled.append((label, re.compile(pat, flags)))

    # accumulators[discipline][year] etc.
    articles = defaultdict(lambda: defaultdict(int))
    tokens = defaultdict(lambda: defaultdict(int))
    raw_min3 = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    once_per_paper = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    once_per_paper_min3 = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    total_items = 0
    matched_items = 0
    refs_stripped_count = 0
    start = time.time()

    with open(jsonl_path, encoding='utf-8') as f:
        for line in f:
            total_items += 1
            if total_items % 100_000 == 0:
                elapsed = time.time() - start
                rate = total_items / elapsed if elapsed > 0 else 0
                strip_pct = (refs_stripped_count / matched_items * 100) if matched_items else 0
                tally = "  ".join(
                    f"{DISCIPLINE_SHORT[d]}: {sum(articles[d].values()):,}"
                    for d in TARGET_DISCIPLINES
                )
                print(
                    f"  {total_items:,} processed ({elapsed:.0f}s, {rate:.0f}/s) "
                    f"[refs stripped: {refs_stripped_count:,} = {strip_pct:.1f}%]\n"
                    f"      {tally}",
                    file=sys.stderr,
                )

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            iid = record.get('iid', '')
            meta = meta_lookup.get(iid)
            if meta is None:
                continue
            year, disc = meta

            combined = ' '.join(record.get('full_text', []))
            combined, was_stripped = strip_references(combined)

            matched_items += 1
            if was_stripped:
                refs_stripped_count += 1

            articles[disc][year] += 1
            tokens[disc][year] += len(WORD_RE.findall(combined))

            for label, pat in compiled:
                n = len(pat.findall(combined))
                if n >= 1:
                    once_per_paper[disc][year][label] += 1
                    if n >= 3:
                        once_per_paper_min3[disc][year][label] += 1
                        raw_min3[disc][year][label] += n

    elapsed = time.time() - start

    # Write 12 CSVs
    mode_buckets = [
        ("raw_min3", raw_min3),
        ("once_per_paper", once_per_paper),
        ("once_per_paper_min3", once_per_paper_min3),
    ]

    for mode_name, bucket in mode_buckets:
        mode_dir = os.path.join(output_dir, mode_name)
        os.makedirs(mode_dir, exist_ok=True)
        for disc in TARGET_DISCIPLINES:
            slug = DISCIPLINE_SLUGS[disc]
            out_path = os.path.join(mode_dir, f"{slug}.csv")
            years = sorted(articles[disc].keys())
            with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['year', 'total_articles', 'total_tokens'] + TERM_LABELS)
                for y in years:
                    row = [y, articles[disc][y], tokens[disc][y]]
                    for label in TERM_LABELS:
                        row.append(bucket[disc][y].get(label, 0))
                    writer.writerow(row)

    # Diagnostic stats
    stats_path = os.path.join(output_dir, 'run_stats.txt')
    strip_pct = (refs_stripped_count / matched_items * 100) if matched_items else 0
    with open(stats_path, 'w') as f:
        f.write("Per-discipline CS-term count run statistics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total JSONL items processed:    {total_items:,}\n")
        f.write(f"Items matched to a target:      {matched_items:,}\n")
        f.write(f"References stripped:            {refs_stripped_count:,} "
                f"({strip_pct:.1f}%)\n")
        f.write(f"Elapsed:                        {elapsed:.1f}s\n\n")
        f.write("Pre-pass metadata classification:\n")
        f.write(f"  Total metadata rows:          {meta_stats['total_metadata_rows']:,}\n")
        f.write(f"  Multi-discipline (skipped):   {meta_stats['multi_discipline']:,}\n")
        f.write(f"  No usable year (skipped):     {meta_stats['no_year']:,}\n")
        for disc in TARGET_DISCIPLINES:
            f.write(f"  {disc:<25}      {per_disc_pre.get(disc, 0):>8,}\n")
        f.write("\nPost-pass article totals (matched in JSONL):\n")
        for disc in TARGET_DISCIPLINES:
            total = sum(articles[disc].values())
            year_min = min(articles[disc]) if articles[disc] else None
            year_max = max(articles[disc]) if articles[disc] else None
            yr_range = f"{year_min}-{year_max}" if year_min is not None else "n/a"
            f.write(f"  {disc:<25} {total:>8,}   years {yr_range}\n")

    # Console summary
    print(f"\nDone in {elapsed:.1f}s", file=sys.stderr)
    print(f"Total items processed: {total_items:,}", file=sys.stderr)
    print(f"Items matched: {matched_items:,}  "
          f"(refs stripped: {refs_stripped_count:,} = {strip_pct:.1f}%)",
          file=sys.stderr)
    for disc in TARGET_DISCIPLINES:
        total = sum(articles[disc].values())
        print(f"  {disc:<25} {total:>8,}", file=sys.stderr)
    print(f"Outputs under: {output_dir}", file=sys.stderr)
    print(f"Stats:         {stats_path}", file=sys.stderr)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.jsonl> <metadata.csv>",
              file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
