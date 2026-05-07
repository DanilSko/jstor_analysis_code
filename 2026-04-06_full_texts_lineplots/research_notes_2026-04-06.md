# JSTOR Full-Text Analysis: Research Notes — 2026-04-06

## Summary

Building on the previous session (2026-03-16), today we produced the first round of actual results: per-year frequency data for 25 CS terms, visualizations, a presentation draft, a reference-removal experiment, and discipline distribution statistics.

---

## 1. Per-Year Term Frequency Data

### Script: `2026-actual-jstor-full-texts/count_cs_terms_by_year.py`

Processes the full 25GB JSONL corpus in a single streaming pass, joining each article to its publication year via the metadata CSV. For each year, counts:
- Total articles
- Total word tokens (alphabetic words only)
- Raw occurrences of each of the 25 CS terms

**Input files:**
- `2026-actual-jstor-full-texts/e119d16a-1426-497d-a42a-a239d354c137.jsonl` (full texts)
- `jstor_analysis_code/jstor_humanities_extended_plus_ling_above_2000_full_metadata.csv` (year lookup via `item_id`)

**Output:** `2026-actual-jstor-full-texts/cs_terms_by_year.csv`
- One row per year (2000--2026), columns: `year, total_articles, total_tokens, <25 term columns>`
- 2026 row has only 143 articles (metadata artifacts); excluded from plots

---

## 2. Visualizations

### Script: `2026-actual-jstor-full-texts/plot_cs_terms_by_year.py`

Reads `cs_terms_by_year.csv` and produces plots in `2026-actual-jstor-full-texts/plots/`:

| File | Description |
|------|-------------|
| `overview_articles_tokens.png` | Bar charts: total articles and total tokens per year (2000--2025) |
| `all_terms_grid.png` | 5x5 small multiples — all 25 terms, relative frequency (per million words) |
| `term_<name>.png` (x25) | Individual line plot per term; y-axis = per-million-word frequency; grey annotations = raw absolute counts |

**Parameters:** `--min-year` and `--max-year` flags (default 2000--2025).

---

## 3. Presentation Draft

### Script: `2026-actual-jstor-full-texts/make_presentation_pdf.py`

Generates a landscape PDF presentation from the plots.

**Output:** `2026-actual-jstor-full-texts/cs_terms_presentation.pdf` (28 slides)

**Slide order:**
1. Title slide (authors, draft date)
2. Corpus overview (articles & tokens bar charts)
3. All-terms grid (small multiples)
4--28. Individual term slides, ordered thematically:
   - General CS terms (Digital, Comput*, Computational, programming)
   - AI/GenAI cluster (AI, Artificial intelligence, Machine learning, ChatGPT, Generative AI, GenAI, Generative Artificial intelligence, Large language model, LLM)
   - NLP-related (Natural language processing, NLP, Named Entity Recognition, Entity Recognition)
   - Methods (Network analysis, Clustering analysis, Pattern recognition, Character recognition, OCR)
   - DH-specific (Distant reading, Humanities Computing, Literary Computing)

---

## 4. Reference Removal Experiment

### Script: `2026-actual-jstor-full-texts/no_references_experiment/count_cs_terms_by_year_no_refs.py`

Same as `count_cs_terms_by_year.py` but strips the references/bibliography section from each article before counting.

**Heuristic for reference removal:**
- Searches for 10 marker patterns: "References", "Bibliography", "Works Cited", "Literature Cited", "References Cited", "Notes and References", "Bibliographie", "Select(ed) Bibliography"
- Markers must appear in the **last 40%** of the text (cutoff at position 0.6)
- Markers must follow a newline or double-space (not mid-sentence)
- Of all matches in the tail region, takes the earliest one and cuts everything from that point onward

**Outputs (in `no_references_experiment/`):**
- `cs_terms_by_year_no_refs.csv` — same format as the original, for direct comparison
- `reference_removal_stats.txt` — diagnostic stats (how many articles had references stripped, strip rate)

**Status:** Script written; awaiting run.

---

## 5. Discipline Statistics

### Script: `jstor_analysis_code/count_disciplines.py`

Parses the metadata CSV and computes:
1. Total papers per discipline
2. Single-discipline papers (papers belonging to only that one discipline)
3. Co-occurrence counts for all discipline pairs

**Outputs (in `jstor_analysis_code/data_output/`):**
- `disciplines_counts.csv` — 68 disciplines, sorted by total papers
- `disciplines_cooccurrence.csv` — 349 pairs, sorted by co-occurrence count

### Key findings

**Top disciplines (of 68 total):**

| Discipline | Total papers | Single-discipline |
|---|---:|---:|
| Language & Literature | 395,031 | 220,739 |
| History | 315,607 | 53,038 |
| American Studies | 99,950 | 0 |
| Religion | 95,743 | 33,988 |
| Art & Art History | 88,437 | 43,437 |
| Philosophy | 88,399 | 38,471 |
| Archaeology | 56,292 | 26,912 |
| Education | 45,921 | 0 |
| Classical Studies | 43,673 | 25,385 |
| Music | 42,485 | 27,340 |
| Linguistics | 28,949 | 8,925 |

**Notable observations:**
- **American Studies, Education, Asian Studies, Business, Security Studies, Irish Studies** etc. always co-occur with another discipline (0 single-discipline papers) — they function as "modifier" disciplines in JSTOR's classification
- **Language & Literature** is by far the largest and most self-contained (56% of its papers are single-discipline)
- **History** is the second largest but much less self-contained (only 17% single-discipline) — it heavily co-occurs with American Studies (89,335 pairs), Religion, Art & Art History, etc.
- **Peace & Conflict Studies + Security Studies** co-occur almost 1:1 (19,791 pairs) — essentially always listed together

**Top co-occurring pairs:**

| Pair | Count |
|---|---:|
| American Studies + History | 89,335 |
| History + Language & Literature | 43,586 |
| Education + Language & Literature | 27,045 |
| History + Religion | 24,029 |
| History + Security Studies | 19,939 |
| Peace & Conflict Studies + Security Studies | 19,791 |

---

## 6. All Scripts Created (Cumulative)

| Script | Location | Purpose |
|--------|----------|---------|
| `count_cs_terms.py` | `2026-actual-jstor-full-texts/` | Count 25 CS terms corpus-wide; outputs CSV |
| `count_cs_terms_by_year.py` | `2026-actual-jstor-full-texts/` | Count 25 CS terms per year + total tokens; outputs CSV |
| `plot_cs_terms_by_year.py` | `2026-actual-jstor-full-texts/` | Plot overview + per-term relative frequency line plots |
| `make_presentation_pdf.py` | `2026-actual-jstor-full-texts/` | Generate landscape PDF presentation from plots |
| `count_cs_terms_by_year_no_refs.py` | `2026-actual-jstor-full-texts/no_references_experiment/` | Same as by-year count but with reference section stripped |
| `count_disciplines.py` | `jstor_analysis_code/` | Discipline counts, single-discipline counts, co-occurrence pairs |
| `train_word2vec.py` | `jstor_analysis_code/` | Train Word2Vec on corpus (not yet run) |

---

## 7. Next Steps

- **Run the reference-removal experiment** and compare `cs_terms_by_year_no_refs.csv` to `cs_terms_by_year.csv` — do the trends change meaningfully?
- **Break down term frequencies by discipline** (not just by year) — which of the 14 humanities disciplines adopt CS terms first/most?
- **Run Word2Vec training** (`train_word2vec.py`) for semantic neighborhood analysis
- **Refine the term list** based on initial results (some terms may be too noisy or too rare)
- Consider the distinction between "pure humanities" vs. "extended humanities" papers for the CS term analysis (RQ3)
