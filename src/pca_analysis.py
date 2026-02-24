import glob
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from sklearn.decomposition import PCA
import wfdb


# -------------------------
# Load ECG files
# -------------------------

DATA_DIR = "/Users/parisa/Desktop/mit-bih-arrhythmia-database"
listNames = glob.glob(f"{DATA_DIR}/*.dat")

if not listNames:
    raise SystemExit("No ECG files found. Check DATA_DIR path in src/pca_analysis.py")
else:
    print(f"Processing {len(listNames)} files.")


# -------------------------
# Filter settings
# -------------------------

fs = 360
fl = 0.5
fh = 20

b1, a1 = signal.butter(N=3, Wn=fl / (fs / 2), btype="high")
b2, a2 = signal.butter(N=3, Wn=fh / (fs / 2), btype="low")


# -------------------------
# Label categories
# -------------------------

table = {
    "N": ["N"],
    "S": ["A", "a", "S", "J"],
    "F": ["F"],
    "V": ["V"],
    "U": ["Q", "U", "/", "p"],
}

data0, data1, data2, data3, data4 = [], [], [], [], []


# -------------------------
# Segmentation
# -------------------------

LEFT = 80
RIGHT = 120

for name in listNames:
    recordName = name.partition(".")[0]
    print(f"Processing record: {recordName}")

    sig, fields = wfdb.rdsamp(record_name=recordName)

    # filtering
    sig = signal.filtfilt(b1, a1, sig[:, 0], padtype="even", axis=0)
    sig = signal.filtfilt(b2, a2, sig, padtype="even")

    # annotations
    annotations = wfdb.rdann(recordName, "atr")
    pos = annotations.sample
    label = annotations.symbol

    # segment beats
    for i in range(1, len(pos) - 1):
        start = pos[i] - LEFT
        end = pos[i] + RIGHT

        # safety check (avoid out-of-range indexing)
        if start < 0 or end > len(sig):
            continue

        qrs = sig[start:end]
        beat_type = label[i]

        if beat_type in table["N"]:
            data0.append(qrs)
        elif beat_type in table["S"]:
            data1.append(qrs)
        elif beat_type in table["F"]:
            data2.append(qrs)
        elif beat_type in table["V"]:
            data3.append(qrs)
        elif beat_type in table["U"]:
            data4.append(qrs)

data0 = np.array(data0, dtype=np.float64)
data1 = np.array(data1, dtype=np.float64)
data2 = np.array(data2, dtype=np.float64)
data3 = np.array(data3, dtype=np.float64)
data4 = np.array(data4, dtype=np.float64)

print("Segmentation is done!")
print("Shapes before limiting:")
print(
    "N:",
    data0.shape,
    "S:",
    data1.shape,
    "F:",
    data2.shape,
    "V:",
    data3.shape,
    "U:",
    data4.shape,
)


# -------------------------
# Limit to max 800 samples (safe)
# -------------------------

data0 = data0[: min(800, len(data0)), :]
data1 = data1[: min(800, len(data1)), :]
data2 = data2[: min(800, len(data2)), :]
data3 = data3[: min(800, len(data3)), :]
data4 = data4[: min(800, len(data4)), :]

print("Shapes after limiting:")
print(
    "N:",
    data0.shape,
    "S:",
    data1.shape,
    "F:",
    data2.shape,
    "V:",
    data3.shape,
    "U:",
    data4.shape,
)


# -------------------------
# PCA
# -------------------------

# Make sure we have some data
if len(data0) == 0:
    raise SystemExit("No N beats found after segmentation. PCA cannot proceed.")

data = np.concatenate((data0, data1, data2, data3, data4), axis=0)

pca = PCA(n_components=60)
pca.fit(data)

print("n_samples:", data.shape[0], "n_features:", data.shape[1])
print("n_components:", pca.n_components_)
print(
    "Total explained variance (full precision):",
    float(np.sum(pca.explained_variance_ratio_)),
)
print("Last cumulative value:", float(np.cumsum(pca.explained_variance_ratio_)[-1]))

# -------------------------
# Explained Variance Analysis (important for Data Analytics)
# -------------------------

explained_variance = pca.explained_variance_ratio_
total_variance = np.sum(explained_variance)

print("-" * 40)
print(f"Total explained variance with 60 components: {total_variance:.4f}")
print("-" * 40)

cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(6, 4))
plt.plot(cumulative_variance)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.grid()
plt.show()

# transform each class
feature0 = pca.transform(data0)
feature1 = pca.transform(data1)
feature2 = pca.transform(data2)
feature3 = pca.transform(data3)
feature4 = pca.transform(data4)


# -------------------------
# Plot PCA (PC1 vs PC2)
# -------------------------

plt.figure(figsize=(10, 7))

plt.plot(feature0[:, 0], feature0[:, 1], "or", label="N")
plt.plot(feature1[:, 0], feature1[:, 1], "ob", label="S")
plt.plot(feature2[:, 0], feature2[:, 1], "xg", label="F")
plt.plot(feature3[:, 0], feature3[:, 1], "vc", label="V")
plt.plot(feature4[:, 0], feature4[:, 1], "ok", label="U")

plt.title("PCA Projection (PC1 vs PC2)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()
