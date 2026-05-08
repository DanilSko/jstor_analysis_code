# JSTOR Full-Text Analysis: Research Notes — 2026-05-08

## Summary

Building on the 2026-04-06 round (corpus-wide per-year CS-term counts), today's session produced **per-discipline** counts for four single-discipline subsets — `Language & Literature`, `Linguistics`, `History`, `Archaeology` — under three different definitions of "what counts as a mention." Outputs: 12 CSVs and 75 line plots. Along the way we found and fixed a latent bug in the reference-stripping regex that had silently been a no-op on this corpus.

---

## 1. Per-Discipline Term Counts

### Script: `2026-actual-jstor-full-texts/per_discipline_counts/count_cs_terms_by_discipline_year.py`

A single streaming pass over the full 25 GB JSONL corpus that filters to single-discipline papers in the four target disciplines and produces three count-mode CSVs per discipline.

**Filter logic.** `discipline_names` (from `jstor_humanities_extended_plus_ling_above_2000_full_metadata.csv`) is parsed with `ast.literal_eval`. A paper is kept iff the unique-disciplines list has length 1 AND that single discipline ∈ targets. Multi-discipline papers and target-irrelevant papers are dropped during a metadata pre-pass — the streaming pass only touches the ~310k retained items.

**Three count modes (per term per year):**

| Mode | What is counted |
|---|---|
| `raw_min3` | Sum of in-paper occurrences, but only for papers where the term appears ≥ 3 times. Papers with 1–2 mentions contribute 0. |
| `once_per_paper` | 1 per paper that contains the term at least once. |
| `once_per_paper_min3` | 1 per paper where the term appears ≥ 3 times. |

For each article we strip the references section, count word tokens, then for each term compute `n = len(pattern.findall(text))`. If `n ≥ 1` we increment `once_per_paper`; if `n ≥ 3` we increment `once_per_paper_min3` and add `n` to `raw_min3`.

**Inputs:**
- `2026-actual-jstor-full-texts/e119d16a-1426-497d-a42a-a239d354c137.jsonl` (full texts)
- `jstor_analysis_code/jstor_humanities_extended_plus_ling_above_2000_full_metadata.csv` (year + discipline lookup via `item_id`)

**Outputs:** 12 CSVs under `2026-actual-jstor-full-texts/per_discipline_counts/`:

```
per_discipline_counts/
├── count_cs_terms_by_discipline_year.py
├── run_stats.txt
├── raw_min3/
│   ├── language_and_literature.csv
│   ├── linguistics.csv
│   ├── history.csv
│   └── archaeology.csv
├── once_per_paper/                    (4 files, same names)
└── once_per_paper_min3/                (4 files, same names)
```

Each CSV has the same column layout as `cs_terms_by_year.csv`: `year, total_articles, total_tokens, <25 term columns>`. `total_articles` and `total_tokens` are the same across all three modes for a given discipline-year (they reflect the post-strip corpus exposure for that discipline) and serve as the denominator for the plotting step.

**Run results** (`run_stats.txt`):

| Discipline | Single-discipline (metadata) | Counted (matched in JSONL) | Δ |
|---|---:|---:|---:|
| Language & Literature | 220,739 | 220,732 | −7 |
| Linguistics | 8,925 | 8,924 | −1 |
| History | 53,038 | 53,036 | −2 |
| Archaeology | 26,912 | 26,910 | −2 |
| **Total processed** | | **309,602** of 1,049,373 JSONL records | |

The 1–7-item drops are records that failed JSON decode or had no `full_text`. Article totals match the published metadata (cross-checked against `disciplines_counts.csv` from 2026-04-06) within 0.01%.

Reference-stripping rate: **12.0% overall** (37,208 / 309,602), **39.2% on >30 k-char articles**. Full-pass elapsed: 1914.6 s (~32 min on Darwin laptop, ~548 items/sec).

---

## 2. Reference-Stripping Bug Fix

### Bug

The existing `2026-actual-jstor-full-texts/no_references_experiment/count_cs_terms_by_year_no_refs.py` had a `REFERENCE_RE` whose every alternative required either `\n` or `\s{2,}` around the marker keyword. But the JSONL stores `full_text` as a list of paragraphs which the script joins with a **single space** — no newlines, no double spaces survive. So the regex matched **nothing**: 0% strip rate. The 2026-04-06 notes say the no-refs script was "awaiting run", which is why the latent bug had never surfaced.

### Fix (used in the new per-discipline script)

Heuristic redesigned for the joined-paragraph form. Instead of relying on whitespace anchors that don't exist, we use a **citation-pattern lookahead** to discriminate a section heading from an in-body mention:

```python
_MARKER = (
    r'(?P<marker>(?i:notes\s+and\s+references|select(?:ed)?\s+bibliography'
    r'|references\s+cited|literature\s+cited|works\s+cited'
    r'|references|bibliography|bibliographie))'
)
_CITE = r"(?:[A-Z][\w\-'À-ſ]+[,.]|\d+\s*\.|[A-Z]\.\s+[A-Z])"
REFERENCE_RE = re.compile(rf'\b{_MARKER}\s+{_CITE}')
```

The marker word(s) must be *immediately* followed by a citation-start pattern: a Capitalized-Word ending in a comma or period (most author surnames), a numbered-list `\d+\.`, or an initial-and-name `A. Smith`. In-body uses ("references to her care", "for bibliography on previous suggestions", "References Smith (1990) found that…") fail the lookahead because they're followed by lowercase function words or by parens / non-comma chars.

Validated on 24 hand-curated test strings (10 TPs / 14 FPs from a manual scan of real corpus tails): 100% TP / 0% FP. The 0.6 cutoff (marker must appear in the last 40% of the article) is kept as additional insurance.

Result: 0% → 12.0% strip rate on the actual corpus. Most unstripped long articles are creative essays without references, archaeology pieces ending in figure captions, or articles ending with a journal-footer page-number block. The heuristic can't strip what the OCR'd text doesn't mark.

> **Note:** the original `count_cs_terms_by_year_no_refs.py` was *not* re-run with the fixed regex in this session — only the new per-discipline script uses the corrected heuristic. If we want a corrected corpus-wide no-refs CSV, that's a one-line edit in the existing no-refs script.

---

## 3. Per-Discipline Line Plots

### Script: `2026-actual-jstor-full-texts/per_discipline_counts/plot_per_discipline.py`

For each of the 3 count modes and each of the 25 CS terms, produces a line plot with 4 lines (one per target discipline) showing per-year prevalence.

**Mode-appropriate normalization** (corrected from initial attempt that used per-million-tokens for all three):

| Mode | Denominator | Y-axis |
|---|---|---|
| `raw_min3` | `total_tokens` | Occurrences per million tokens |
| `once_per_paper` | `total_articles` | Percentage of papers |
| `once_per_paper_min3` | `total_articles` | Percentage of papers |

The first round used `count / total_tokens` for all three modes, which conflated paper-level engagement with article-length differences (Linguistics articles tend to be shorter than History ones, so the same paper-level mention rate showed up at different y-values). Switching the paper-count modes to `count / total_articles × 100` makes the four disciplines directly comparable on "what fraction of papers talk about this term."

**Plot styling:** four lines with fixed colors — L&L blue, Linguistics green, History red, Archaeology orange — markers + lines, in-plot legend, year range 2000–2025 (2026 dropped per existing convention).

**Output:** `2026-actual-jstor-full-texts/per_discipline_counts/plots/<mode>/term_<safe_name>.png`, 3 × 25 = **75 plots**.

```
plots/
├── raw_min3/                 (25 PNGs)
├── once_per_paper/           (25 PNGs)
└── once_per_paper_min3/      (25 PNGs)
```

CLI flags: `--min-year` / `--max-year` (default 2000–2025).

---

## 4. Verification (cross-checks performed)

1. Headers of all 12 CSVs are byte-identical to `cs_terms_by_year.csv` (28 cols, 25 terms).
2. `total_articles` and `total_tokens` are consistent across the 3 modes within each discipline (same denominators — sanity that the streaming pass populated them once and only once).
3. Cross-mode invariant: for every (discipline, year, term) cell, `once_per_paper ≥ once_per_paper_min3` AND `raw_min3 ≥ 3 × once_per_paper_min3`. **0 violations** across all ~7,800 cells.
4. Article totals per year, summed across the four target disciplines, are stable at ~28–30% of corpus-wide article counts (309,602 / 1,049,373 = 29.5%) — consistent with the published metadata.
5. Visual spot-checks on `term_Digital.png` (gradual rise, sharp uptick after 2020) and `term_ChatGPT.png` (flat zero pre-2023, sharp rise after, L&L leading) pass the smell test.

---

## 5. Scripts Created This Session

| Script | Location | Purpose |
|--------|----------|---------|
| `count_cs_terms_by_discipline_year.py` | `2026-actual-jstor-full-texts/per_discipline_counts/` | Single streaming pass — produces 12 CSVs + run_stats.txt |
| `plot_per_discipline.py` | `2026-actual-jstor-full-texts/per_discipline_counts/` | 75 per-mode-per-term comparison line plots |

Both copy (rather than import) the relevant constants from the existing `count_cs_terms_by_year_no_refs.py` and `count_disciplines.py`, matching the project's standalone-script convention.

---

## 6. Next Steps

- Decide whether to re-run the corpus-wide no-refs script with the fixed regex (and regenerate `cs_terms_by_year_no_refs.csv` + dependent plots).
- Inspect the 75 per-discipline plots and pick the most striking trends for the presentation. Likely candidates: AI cluster (`AI`, `Artificial intelligence`, `Machine learning`, `ChatGPT`), `Digital`, `Computational`, `Distant reading` (L&L-specific?), `Network analysis` (History/Archaeology?).
- Consider per-discipline grids (5×5 small multiples per discipline, like the existing `all_terms_grid.png` but per-discipline) if individual term plots prove unwieldy.
- Optionally add per-discipline overview plots (articles + tokens per year) so the small-denominator volatility (e.g. Linguistics late-year spikes) is visible without having to read it off the CSV.
