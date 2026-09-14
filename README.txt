=============================================================================
SMART-GRID FAULT DETECTION — THESIS PROJECT PACKAGE
=============================================================================
Thesis : Implementation of a Machine Learning Technique for Fault
         Detection in Smart-Grid Systems
Author : Bright
Year   : 2026
=============================================================================

FOLDER CONTENTS
=============================================================================

📁 thesis_documents/
   All thesis chapters and supporting documents:

   Thesis_Corrected.docx           — Full corrected thesis (all 5 chapters)
                                      with all 16 supervisor corrections applied.
                                      Replace [From pipeline output] placeholders
                                      by running the training code first.

   Chapter3_Methodology.docx       — Chapter 3: Corrected Methodology
   Chapter4_Updated.docx           — Chapter 4: Results (actual trained results,
                                      no relay baselines, 8 figures embedded)
   Chapter4_5_Combined.docx        — Chapter 4 + 5 combined (condensed Ch.5)
   Chapter4_WithFigures.docx       — Chapter 4 with full 12-figure set
   Chapter5_Conclusions.docx       — Chapter 5: Conclusions & Recommendations
   Abbreviations.docx              — List of Abbreviations (70 entries, A3)
   LitReview_STRIP.docx            — Chapter 2: Literature Review (STRIP)
   References_STRIP_30papers.docx  — 30 annotated references (APA 7)
   Research_Proposal_Outline.docx  — Original research proposal outline
   Research_Proposal_Detailed.docx — Detailed research proposal
   PS_IDEAL_halfpage.docx          — Problem statement (IDEAL framework)

📁 correction_documents/
   Supervisor review response documents:

   Thesis_Correction_Guide.docx       — Step-by-step guide to all 16 corrections
   Priority_Revision_Register.docx    — Table 1: Priority Revision Register
                                         (C1–C7, M1–M6, S1–S3 response table,
                                          A3 landscape, colour-coded)

📁 code/
   Python training pipeline:

   smart_grid_training.py    — MAIN FILE: Complete training pipeline.
                                Trains RF and HistGradientBoostingClassifier.
                                Run: python smart_grid_training.py --data classData.csv
                                Demo: python smart_grid_training.py

   corrected_pipeline.py     — Alternative full pipeline with extra documentation
   verify_dataset.py         — Verify classData.csv before training
   requirements.txt          — Python package requirements

=============================================================================
HOW TO RUN THE TRAINING CODE
=============================================================================

1. Install Python 3.10+ from python.org

2. Install packages:
      pip install -r requirements.txt

3. Download classData.csv from Kaggle:
      https://www.kaggle.com/datasets/esathyaprakash/
      electrical-fault-detection-and-classification

4. Verify your dataset:
      python verify_dataset.py --data classData.csv

5. Run the full training pipeline:
      python smart_grid_training.py --data classData.csv

6. Copy metric values from output/training_log.txt into the thesis.
   Replace all [From pipeline output] placeholders in Thesis_Corrected.docx.

=============================================================================
SUPERVISOR CORRECTIONS APPLIED
=============================================================================
C1  Class counts from code — not typed manually
C2  Split FIRST before SMOTE (data leakage fixed)
C3  Dataset fully identified with APA 7 citation
C4  All metrics from saved y_pred files — not entered manually
C5  HistGradientBoostingClassifier (not GradientBoostingClassifier)
C6  Relay baselines removed from scope; objectives revised
C7  Reproducibility package: y_pred, split indices, pipeline objects
M1  Scalar row structure clarified (no window grouping issue)
M2  "Systematic" → "structured narrative" review
M3  APA 7 citations corrected throughout
M4  Promised vs delivered content reconciled
M5  Bootstrap CI and McNemar's test added
M6  Cross-study comparison qualified
S1  Heading numbers corrected throughout
S2  Objectives aligned with actual experiment
S3  Grammar, spelling, British English corrected

=============================================================================
DATASET CITATION (APA 7)
=============================================================================
Sathyaprakash, E. (2021). Electrical fault detection and classification
[Dataset]. Kaggle. https://www.kaggle.com/datasets/esathyaprakash/
electrical-fault-detection-and-classification
=============================================================================
