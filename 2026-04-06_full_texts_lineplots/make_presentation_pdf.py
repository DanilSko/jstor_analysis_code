#!/usr/bin/env python3
"""
Generate a landscape PDF presentation of the CS-terms-in-JSTOR charts.

Usage:
    python make_presentation_pdf.py

Output:
    cs_terms_presentation.pdf
"""
import os
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
)

PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'cs_terms_presentation.pdf')

PAGE_W, PAGE_H = landscape(letter)  # 11 x 8.5 inches


def build_styles():
    base = getSampleStyleSheet()
    styles = {
        'title': ParagraphStyle(
            'SlideTitle', parent=base['Title'],
            fontSize=28, leading=34, alignment=TA_CENTER,
            textColor=HexColor('#1a3a5c'), spaceAfter=12,
        ),
        'subtitle': ParagraphStyle(
            'SlideSubtitle', parent=base['Normal'],
            fontSize=16, leading=20, alignment=TA_CENTER,
            textColor=HexColor('#555555'), spaceAfter=6,
        ),
        'heading': ParagraphStyle(
            'SlideHeading', parent=base['Heading1'],
            fontSize=22, leading=28, alignment=TA_CENTER,
            textColor=HexColor('#1a3a5c'), spaceBefore=0, spaceAfter=8,
        ),
        'body': ParagraphStyle(
            'SlideBody', parent=base['Normal'],
            fontSize=13, leading=18, alignment=TA_LEFT,
            textColor=HexColor('#333333'),
        ),
        'caption': ParagraphStyle(
            'Caption', parent=base['Normal'],
            fontSize=10, leading=13, alignment=TA_CENTER,
            textColor=HexColor('#777777'),
        ),
    }
    return styles


def slide_image(story, title, image_path, caption, styles, img_height=5.2 * inch):
    """Add a slide with a heading, a centered image, and a caption."""
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(title, styles['heading']))

    img = Image(image_path)
    # Scale to fit: max width = page minus margins, max height = img_height
    max_w = PAGE_W - 2 * inch
    aspect = img.imageWidth / img.imageHeight
    w = min(max_w, img_height * aspect)
    h = w / aspect
    img.drawWidth = w
    img.drawHeight = h
    img.hAlign = 'CENTER'
    story.append(img)

    if caption:
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph(caption, styles['caption']))

    story.append(PageBreak())


def main():
    styles = build_styles()
    story = []

    # --- Title slide ---
    story.append(Spacer(1, 2.0 * inch))
    story.append(Paragraph(
        'CS Terms in JSTOR Humanities Publications', styles['title']))
    story.append(Paragraph(
        'Preliminary exploration of ~25 key terms across 1,049,373 full-text articles (2000-2025)',
        styles['subtitle']))
    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph(
        'Vered Silber-Varod, Daniil Skorinkin, Nitza Geri',
        styles['subtitle']))
    story.append(Paragraph('Draft -- April 2026', styles['caption']))
    story.append(PageBreak())

    # --- Overview slide ---
    slide_image(
        story,
        'Corpus overview: articles and tokens per year',
        os.path.join(PLOTS_DIR, 'overview_articles_tokens.png'),
        'JSTOR Humanities (extended) full-text corpus, English, 2000-2025. '
        'Note the drop after ~2020 reflects JSTOR ingestion lag, not a real decline.',
        styles,
    )

    # --- All terms grid ---
    slide_image(
        story,
        'All 25 CS terms: relative frequency (per million words)',
        os.path.join(PLOTS_DIR, 'all_terms_grid.png'),
        'Each subplot shows occurrences per million words for one search term.',
        styles,
        img_height=5.8 * inch,
    )

    # --- Individual term slides ---
    term_files = [
        ('Digital', 'term_Digital.png'),
        ('Comput*', 'term_Comput_star.png'),
        ('Computational', 'term_Computational.png'),
        ('programming', 'term_programming.png'),
        ('AI', 'term_AI.png'),
        ('Artificial intelligence', 'term_Artificial_intelligence.png'),
        ('Machine learning', 'term_Machine_learning.png'),
        ('ChatGPT', 'term_ChatGPT.png'),
        ('Generative AI', 'term_Generative_AI.png'),
        ('GenAI', 'term_GenAI.png'),
        ('Generative Artificial intelligence', 'term_Generative_Artificial_intelligence.png'),
        ('Large language model', 'term_Large_language_model.png'),
        ('LLM', 'term_LLM.png'),
        ('Natural language processing', 'term_Natural_language_processing.png'),
        ('NLP', 'term_NLP.png'),
        ('Named Entity Recognition', 'term_Named_Entity_Recognition.png'),
        ('Entity Recognition', 'term_Entity_Recognition.png'),
        ('Network analysis', 'term_Network_analysis.png'),
        ('Clustering analysis', 'term_Clustering_analysis.png'),
        ('Pattern recognition', 'term_Pattern_recognition.png'),
        ('Character recognition', 'term_Character_recognition.png'),
        ('OCR', 'term_OCR.png'),
        ('Distant reading', 'term_Distant_reading.png'),
        ('Humanities Computing', 'term_Humanities_Computing.png'),
        ('Literary Computing', 'term_Literary_Computing.png'),
    ]

    for term_label, filename in term_files:
        slide_image(
            story,
            f'"{term_label}"',
            os.path.join(PLOTS_DIR, filename),
            None,
            styles,
        )

    # --- Build PDF ---
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=landscape(letter),
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    doc.build(story)
    print(f"Presentation saved to: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
