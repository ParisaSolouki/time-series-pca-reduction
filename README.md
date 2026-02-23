# Dimensionality Reduction on High-Dimensional Time-Series Data using PCA

## Overview
This project demonstrates an end-to-end workflow for transforming high-dimensional time-series segments into a lower-dimensional feature space using Principal Component Analysis (PCA).

Although the source data consists of ECG time-series segments, the primary focus of this project is on:
- Feature structuring
- Dimensionality reduction
- Variance preservation
- Visualization of separability in feature space

These techniques are directly transferable to general data analytics and machine learning workflows.

---

## Objective
To reduce high-dimensional segmented time-series data into a compact representation while preserving the most significant variance patterns.

---

## Workflow

### 1. Data Preparation
- Load raw time-series files and annotations
- Segment time-series windows around key events
- Structure segments into a matrix format suitable for analysis

### 2. Preprocessing
- Apply bandpass filtering to reduce noise
- Standardize segments before PCA transformation

### 3. PCA Application
- Fit PCA model on structured segments
- Reduce dimensionality to 60 principal components
- Analyze explained variance ratio
- Transform each segment into reduced feature space

### 4. Visualization
- Plot principal component projections (PC1 vs PC2)
- Assess clustering and separability patterns
- Compare variance distribution across components

---

## Key Results (to be completed with actual values)
- The first X components preserve approximately Y% of total variance.
- Clear separation patterns are observed in the first two principal components.
- Dimensionality reduced from N dimensions to 60 components.

---

## Why This Matters
Dimensionality reduction is critical when working with:
- High-dimensional structured datasets
- Time-series feature matrices
- Large-scale behavioral or sensor data

PCA improves:
- Computational efficiency
- Noise reduction
- Feature interpretability
- Visualization clarity

---

## Tech Stack
- Python
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- wfdb

---

## Future Improvements
- Add variance explained plot
- Compare PCA with alternative reduction techniques
- Export transformed features for downstream modeling

