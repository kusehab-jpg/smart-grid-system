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


📁 code/
   Python training pipeline:

   smart_grid_training.py    — MAIN FILE: Complete training pipeline.
                                Trains RF and HistGradientBoostingClassifier.
                                Run: python smart_grid_training.py --data classData.csv
                                Demo: python smart_grid_training.py

   pipeline.py     — Alternative full pipeline with extra documentation
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

=============================================================================
DATASET CITATION (APA 7)
=============================================================================
Sathyaprakash, E. (2021). Electrical fault detection and classification
[Dataset]. Kaggle. https://www.kaggle.com/datasets/esathyaprakash/
electrical-fault-detection-and-classification
=============================================================================
