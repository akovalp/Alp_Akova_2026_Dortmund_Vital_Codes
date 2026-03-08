# Dortmund Vital Study Analysis

This repository contains the cleaned-up version of my initial Dortmund Vital study analysis. It includes:

- **A)** All extraction logic conducted so far (source power, spectral flatness, connectivity)
- **B)** All machine learning analyses performed on these extracted features

## Project Structure

This section provides descriptions of the main folders in this repository.

### Extraction Codes

For every extraction method, I have implemented the same (or similar) metrics in both sensor and source space. The rationale is to enable one-to-one comparisons of source-derived vs. electrode-derived features in machine learning pipelines.

#### Extraction Based on ROI (Source and Electrode)

This folder contains code for extracting source and electrode power and related metrics from four dedicated regions of interest (ROIs) in the sample. Each subfolder includes a `methods.md` file with a technical overview of the tools and analyses used.

#### Extraction Based on Connectivity

This folder stores wPLI (weighted Phase Lag Index) extracted connectivity metrics for both source and electrode space.

### Notebooks

I extensively use Jupyter notebooks for all analyses, normalization, and dataset construction. The notebooks directory will contain subdirectories with descriptive names, each including documentation of what has been achieved to facilitate testing and further discussion.


### Machine Learning

This directory contains the predictive modeling and feature selection pipelines for estimating chronological age from the extracted EEG features. To enable direct comparisons, the pipelines are duplicated to run on both source-space and sensor-space datasets independently.

The machine learning workflow is structured around two primary components:

#### 1. Robust Feature Selection (Sparse Group Lasso)
- **Scripts:** `ROI_Based_Feature_Selection_ML_age.py`, `ROI_Based_Feature_Selection_ML_age_sensor.py`
- **Overview:** Implements Sparse Group Lasso (SGL) to identify the most stable and predictive features while respecting the inherent hierarchical structure of the data (e.g., grouping by feature type and anatomical region). 
- **Key Mechanics:** 
  - Utilizes stratified nested cross-validation.
  - Applies Complementary Pairs Stability Selection (CPSS) and measures selection robustness using the Nogueira stability index.
  - Benchmarks different structured grouping strategies (Tier 1 feature-type vs. Tier 2 spatial-type) against a plain LASSO baseline.

#### 2. Age Prediction (Custom XGBoost Regressor)
- **Scripts:** `xgboost_combined_directional_cv.py`, `xgboost_combined_directional_cv_sensor.py`
- **Overview:** Employs an XGBoost regressor to predict chronological age using the optimally selected features.
- **Custom Objective Function:** Addresses common brain-age prediction pitfalls by combining two weighting schemes:
  - *Directional Bias Weighting:* Asymmetrically penalizes errors to counteract regression-to-the-mean (e.g., heavily penalizes over-predicting age for young subjects or under-predicting for old subjects).
  - *Inverse-Frequency Weighting:* Gives higher importance to under-represented age groups to ensure balanced learning.
- **Interpretability:** Extracts out-of-fold (OOF) SHAP-style additive contributions to evaluate both global and group-specific feature importance.
