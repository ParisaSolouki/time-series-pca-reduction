# Dimensionality Reduction on ECG Time-Series using PCA

## Project Overview

This project applies **Principal Component Analysis (PCA)** to high-dimensional ECG time-series data from the MIT-BIH Arrhythmia Database.

Each QRS complex consists of 200 time samples, resulting in a high-dimensional feature space.  
PCA is used to:

- Reduce dimensionality
- Analyze variance distribution
- Visualize class separability
- Understand intrinsic structure of ECG patterns

Although the dataset is biomedical, the core focus is on **time-series feature engineering and dimensionality reduction techniques** commonly used in data analytics.

---

## Objectives

- Segment ECG signals around annotated R-peaks
- Extract fixed-length QRS complexes
- Apply PCA for dimensionality reduction
- Analyze cumulative explained variance
- Visualize class separability in reduced feature space

---

## Dataset

**MIT-BIH Arrhythmia Database**

Each ECG record is:

- Filtered using Butterworth high-pass and low-pass filters
- Segmented around R-peaks
- Categorized into five heartbeat classes:

| Label | Description |
|-------|------------|
| N | Normal beats |
| S | Supraventricular beats |
| F | Fusion beats |
| V | Ventricular beats |
| U | Unknown beats |

---

## Methodology

### 1. Signal Preprocessing
- Sampling frequency: 360 Hz
- High-pass filter: 0.5 Hz
- Low-pass filter: 20 Hz
- 3rd-order Butterworth filters

### 2. Segmentation
Each QRS complex is extracted using:
- 80 samples before R-peak
- 120 samples after R-peak  
→ Total: **200-dimensional feature vector**

### 3. Dimensionality Reduction (PCA)

- PCA applied with 60 components
- Model fitted on the full concatenated dataset
- Features transformed per class

### Explained Variance Analysis

- Cumulative explained variance is computed
- With 60 components, nearly 100% variance is retained
- Demonstrates strong redundancy in raw time-series features

This validates PCA as an effective compression technique for ECG signals.

---

## Visualization

The first two principal components are plotted to evaluate:

- Cluster separation between heartbeat classes
- Structural differences in signal morphology
- Potential classification boundaries

---

## Project Structure

```text
time-series-pca-reduction/
│
├── notebooks/
│   └── ECG_Applying_PCA.ipynb
│
├── src/
│   └── pca_analysis.py
│
└── README.md
```