#!/usr/bin/env python3
"""
Per-discipline keyword line plots.

For each count mode (raw_min3, once_per_paper, once_per_paper_min3) and each
of the 25 CS terms, produce a line plot with 4 lines -- one per discipline
(Language & Literature, Linguistics, History, Archaeology) -- showing the
keyword's per-year prevalence in that discipline.

Normalization is mode-appropriate:
  - raw_min3            : count / total_tokens * 1,000,000  (occurrences per
                          million tokens -- it's an occurrence count)
  - once_per_paper      : count / total_articles * 100      (% of papers --
                          it's a paper count)
  - once_per_paper_min3 : count / total_articles * 100

Inputs: the 12 CSVs under raw_min3/, once_per_paper/, once_per_paper_min3/
        produced by count_cs_terms_by_discipline_year.py.

Outputs: plots/<mode>/term_<safe_name>.png  (3 x 25 = 75 plots)

Usage:
    python plot_per_discipline.py
    python plot_per_discipline.py --min-year 2000 --max-year 2025
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(ROOT, 'plots')

MODES = ['raw_min3', 'once_per_paper', 'once_per_paper_min3']

DISCIPLINES = [
    ('Language & Literature', 'language_and_literature.csv'),
    ('Linguistics',           'linguistics.csv'),
    ('History',               'history.csv'),
    ('Archaeology',           'archaeology.csv'),
]

# Colors picked for distinguishability in print and on screen.
DISCIPLINE_COLORS = {
    'Language & Literature': '#1f77b4',  # blue
    'Linguistics':           '#2ca02c',  # green
    'History':               '#d62728',  # red
    'Archaeology':           '#ff7f0e',  # orange
}

TERM_COLUMNS = [
    'programming', 'AI', 'Artificial intelligence', 'Character recognition',
    'ChatGPT', 'Clustering analysis', 'Comput*', 'Computational', 'Digital',
    'Distant reading', 'Entity Recognition', 'GenAI', 'Generative AI',
    'Generative Artificial intelligence', 'Humanities Computing',
    'Large language model', 'LLM', 'Literary Computing', 'Machine learning',
    'Natural language processing', 'NLP', 'Named Entity Recognition',
    'Network analysis', 'OCR', 'Pattern recognition',
]

MODE_TITLES = {
    'raw_min3':            'raw counts (only papers with ≥3 mentions)',
    'once_per_paper':      'one per paper (≥1 mention)',
    'once_per_paper_min3': 'one per paper (≥3 mentions)',
}

# (denominator_column, multiplier, y-axis label) per mode
MODE_NORMALIZATION = {
    'raw_min3':            ('total_tokens',   1_000_000, 'Occurrences per million tokens'),
    'once_per_paper':      ('total_articles', 100,       'Percentage of papers'),
    'once_per_paper_min3': ('total_articles', 100,       'Percentage of papers'),
}


def safe_filename(term):
    return term.replace('*', '_star').replace(' ', '_')


def load_mode(mode, min_year, max_year):
    """Return {discipline: dataframe} for one mode, year-filtered."""
    denom_col, mult, _ = MODE_NORMALIZATION[mode]
    out = {}
    for disc, fname in DISCIPLINES:
        path = os.path.join(ROOT, mode, fname)
        df = pd.read_csv(path)
        df = df[(df['year'] >= min_year) & (df['year'] <= max_year)].copy()
        for term in TERM_COLUMNS:
            df[f'{term}_norm'] = df[term] / df[denom_col] * mult
        out[disc] = df
    return out


def plot_term(term, mode, dfs, out_dir):
    _, _, ylabel = MODE_NORMALIZATION[mode]
    fig, ax = plt.subplots(figsize=(11, 5))
    for disc, df in dfs.items():
        ax.plot(df['year'], df[f'{term}_norm'],
                marker='o', markersize=4, linewidth=1.6,
                color=DISCIPLINE_COLORS[disc], label=disc)

    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel)
    ax.set_title(f'"{term}"  --  {MODE_TITLES[mode]}')
    ax.grid(axis='both', alpha=0.3)
    ax.legend(loc='best', fontsize=9, frameon=True)

    plt.tight_layout()
    path = os.path.join(out_dir, f'term_{safe_filename(term)}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-year', type=int, default=2000)
    parser.add_argument('--max-year', type=int, default=2025)
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)

    for mode in MODES:
        out_dir = os.path.join(PLOTS_DIR, mode)
        os.makedirs(out_dir, exist_ok=True)

        print(f'Loading {mode} ...')
        dfs = load_mode(mode, args.min_year, args.max_year)
        n_years = len(next(iter(dfs.values())))
        print(f'  {n_years} years per discipline; plotting {len(TERM_COLUMNS)} terms')

        for term in TERM_COLUMNS:
            plot_term(term, mode, dfs, out_dir)
        print(f'  -> {out_dir}/')

    total = len(MODES) * len(TERM_COLUMNS)
    print(f'\nDone. {total} plots written under {PLOTS_DIR}/')


if __name__ == '__main__':
    main()
