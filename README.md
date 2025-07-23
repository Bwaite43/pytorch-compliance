# PyTorch Compliance Analysis System

## Overview
This project provides a comprehensive PyTorch-based system for analyzing, mapping, and predicting compliance status for security controls across GovRAMP and NIST frameworks. It is designed to help organizations understand their compliance posture, identify gaps, and generate actionable recommendations using machine learning and data analysis techniques.

## Features
- **Control Similarity Mapping:** Uses neural network embeddings to find similar controls between GovRAMP and NIST frameworks, enabling cross-framework alignment and mapping.
- **Implementation Status Prediction:** Predicts the implementation status of controls using a trained PyTorch classification model.
- **Compliance Gap Analysis:** Identifies gaps in compliance coverage by comparing implemented controls against target framework requirements.
- **Intelligent Recommendations:** Generates recommendations for improving compliance based on control family similarities and coverage analysis.
- **Comprehensive Reporting:** Outputs a detailed JSON report summarizing mappings, gap analysis, and recommendations.

## How It Works
1. **Data Loading:** Loads GovRAMP and NIST control datasets from CSV files, standardizes columns, and combines them for analysis.
2. **Feature Engineering:** Extracts text and categorical features for each control, including TF-IDF vectors and one-hot encodings.
3. **Model Training:**
   - Trains a neural network to generate control embeddings for similarity analysis.
   - Trains a classification model to predict implementation status.
4. **Analysis & Reporting:**
   - Finds similar controls across frameworks.
   - Maps controls between GovRAMP and NIST.
   - Analyzes compliance gaps and generates recommendations.
   - Saves results and models to disk.

## Usage
Run the main script to perform the full analysis:

```bash
python pytorch/pytorch_compliance.py
```

### Output Files
- `best_embedding_model.pth`: Trained control embedding model.
- `best_status_model.pth`: Trained status prediction model.
- `compliance_report.json`: Comprehensive analysis report.

## Requirements
- Python 3.7+
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Data Files
Place the following files in the appropriate directories:
- `data-convert/GovRAMPCoreControls_manual.csv`: GovRAMP controls data
- `Data/NIST_SP-800-53_rev5_catalog_load.csv`: NIST controls data

## Project Structure
```
README.md
requirements.txt
Data/
    NIST_SP-800-53_rev5_catalog_load.csv
...
data-convert/
    GovRAMPCoreControls_manual.csv
...
pytorch/
    pytorch_compliance.py
```

## License
This project is provided for educational and research purposes. See repository for license details.
# pytorch-compliance