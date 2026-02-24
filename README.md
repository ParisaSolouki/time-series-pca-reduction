# Dimensionality Reduction on ECG Time-Series using PCA

## 📌 Project Overview

This project applies **Principal Component Analysis (PCA)** to high-dimensional ECG time-series data from the MIT-BIH Arrhythmia Database.

The goal is to:
- Segment ECG signals around R-peaks
- Extract QRS complexes
- Reduce dimensionality using PCA
- Visualize class separability in low-dimensional space

---

## 📂 Dataset

Dataset used:
MIT-BIH Arrhythmia Database

Each ECG signal is:
- Filtered using Butterworth high-pass and low-pass filters
- Segmented around annotated R-peaks
- Categorized into five heartbeat classes:

| Label | Description |
|-------|------------|
| N | Normal beats |
| S | Supraventricular beats |
| F | Fusion beats |
| V | Ventricular beats |
| U | Unknown beats |

---

## ⚙️ Methodology

### 1️⃣ Signal Preprocessing
- Sampling frequency: 360 Hz
- High-pass filter: 0.5 Hz
- Low-pass filter: 20 Hz
- Butterworth filters (3rd order)

### 2️⃣ Segmentation
Each QRS complex is extracted using a window:
- 80 samples before R-peak
- 120 samples after R-peak

### 3️⃣ Dimensionality Reduction
- PCA with 60 components
- Data concatenated across classes
- PCA fitted on full dataset
- Features transformed per class

---

## 📊 PCA Visualization

The first two principal components are plotted to visualize class distribution.

This helps evaluate how well different ECG classes are separable in reduced-dimensional space.

---

## 🗂️ Project Structure

```
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
---

## ▶️ How to Run

Clone the repository:

```bash
git clone https://github.com/ParisaSolouki/time-series-pca-reduction.git
cd time-series-pca-reduction
```