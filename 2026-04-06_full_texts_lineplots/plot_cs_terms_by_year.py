#!/usr/bin/env python3
"""
Plot per-year trends from cs_terms_by_year.csv:
  1. Total articles per year
  2. Total word tokens per year
  3. One line plot per CS term showing relative frequency (per million words)

Usage:
    python plot_cs_terms_by_year.py
    python plot_cs_terms_by_year.py --min-year 2000 --max-year 2025

Outputs PNG files into a plots/ subfolder.
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

# --- Configuration ---

CSV_PATH = os.path.join(os.path.dirname(__file__), 'cs_terms_by_year.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'plots')

TERM_COLUMNS = [
    'programming', 'AI', 'Artificial intelligence', 'Character recognition',
    'ChatGPT', 'Clustering analysis', 'Comput*', 'Computational', 'Digital',
    'Distant reading', 'Entity Recognition', 'GenAI', 'Generative AI',
    'Generative Artificial intelligence', 'Humanities Computing',
    'Large language model', 'LLM', 'Literary Computing', 'Machine learning',
    'Natural language processing', 'NLP', 'Named Entity Recognition',
    'Network analysis', 'OCR', 'Pattern recognition',
]


def plot_overview(df, output_dir):
    """Plot total articles and total tokens per year (two subplots)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.bar(df['year'], df['total_articles'], color='steelblue', alpha=0.8)
    ax1.set_ylabel('Articles')
    ax1.set_title('Total articles per year')
    ax1.grid(axis='y', alpha=0.3)
    # Add value labels on bars
    for _, row in df.iterrows():
        ax1.text(row['year'], row['total_articles'], f"{row['total_articles']:,.0f}",
                 ha='center', va='bottom', fontsize=6, rotation=45)

    ax2.bar(df['year'], df['total_tokens'] / 1e6, color='darkorange', alpha=0.8)
    ax2.set_ylabel('Tokens (millions)')
    ax2.set_title('Total word tokens per year')
    ax2.set_xlabel('Year')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'overview_articles_tokens.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_single_term(df, term, output_dir):
    """Line plot of relative frequency (per million words) for one term."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df['year'], df[f'{term}_pmw'], marker='o', markersize=4,
            linewidth=1.5, color='steelblue')
    ax.fill_between(df['year'], df[f'{term}_pmw'], alpha=0.1, color='steelblue')

    ax.set_xlabel('Year')
    ax.set_ylabel('Occurrences per million words')
    ax.set_title(f'"{term}" — relative frequency in JSTOR Humanities full texts')
    ax.grid(axis='both', alpha=0.3)

    # Show raw counts as secondary info
    for _, row in df.iterrows():
        raw = row[term]
        if raw > 0:
            ax.annotate(f'{int(raw)}', (row['year'], row[f'{term}_pmw']),
                        textcoords='offset points', xytext=(0, 8),
                        fontsize=6, ha='center', color='grey')

    plt.tight_layout()
    # Sanitize filename
    safe_name = term.replace('*', '_star').replace(' ', '_')
    path = os.path.join(output_dir, f'term_{safe_name}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_all_terms_grid(df, output_dir):
    """All 25 terms as small multiples in a single figure."""
    n = len(TERM_COLUMNS)
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(24, rows * 3.2))
    axes = axes.flatten()

    for i, term in enumerate(TERM_COLUMNS):
        ax = axes[i]
        ax.plot(df['year'], df[f'{term}_pmw'], marker='.', markersize=3,
                linewidth=1.2, color='steelblue')
        ax.fill_between(df['year'], df[f'{term}_pmw'], alpha=0.1, color='steelblue')
        ax.set_title(term, fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)
        ax.grid(axis='both', alpha=0.3)

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.supxlabel('Year', fontsize=11)
    fig.supylabel('Occurrences per million words', fontsize=11)
    fig.suptitle('CS terms in JSTOR Humanities publications — relative frequency over time',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0.02, 0.02, 1, 0.96])

    path = os.path.join(output_dir, 'all_terms_grid.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-year', type=int, default=2000)
    parser.add_argument('--max-year', type=int, default=2025)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    df = pd.read_csv(CSV_PATH)
    df = df[(df['year'] >= args.min_year) & (df['year'] <= args.max_year)]
    print(f"Loaded {len(df)} years ({df['year'].min()}--{df['year'].max()})")

    # Compute per-million-word frequencies
    for term in TERM_COLUMNS:
        df[f'{term}_pmw'] = df[term] / df['total_tokens'] * 1_000_000

    # 1. Overview plot
    print("\nPlotting overview...")
    plot_overview(df, OUTPUT_DIR)

    # 2. Individual term plots
    print("\nPlotting individual terms...")
    for term in TERM_COLUMNS:
        plot_single_term(df, term, OUTPUT_DIR)

    # 3. All terms grid
    print("\nPlotting all-terms grid...")
    plot_all_terms_grid(df, OUTPUT_DIR)

    print(f"\nDone. All plots saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
