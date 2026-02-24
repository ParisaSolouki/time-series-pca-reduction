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
    print("No ECG files found in the specified directory.")
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

data0 = []
data1 = []
data2 = []
data3 = []
data4 = []


# -------------------------
# Segmentation
# -------------------------

for name in listNames:
    parts = name.partition(".")
    recordName = parts[0]
    print(recordName)

    sig, fields = wfdb.rdsamp(record_name=recordName)

    sig = signal.filtfilt(b1, a1, sig[:, 0], padtype="even", axis=0)
    sig = signal.filtfilt(b2, a2, sig, padtype="even")

    annotations = wfdb.rdann(recordName, "atr")
    pos = annotations.sample
    label = annotations.symbol

    fs = fields["fs"]

    for i in range(1, len(pos) - 1):
        indx = list(range(pos[i] - 80, pos[i] + 120))
        qrs = sig[indx]
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


# -------------------------
# Limit to 800 samples
# -------------------------

data0 = data0[:800, :]
data1 = data1[:800, :]
data2 = data2[:800, :]
data3 = data3[:800, :]
data4 = data4[:800, :]


# -------------------------
# PCA
# -------------------------

data = np.concatenate((data0, data1, data2, data3, data4), axis=0)

pca = PCA(n_components=60)
pca.fit(data)

feature0 = pca.transform(data0)
feature1 = pca.transform(data1)
feature2 = pca.transform(data2)
feature3 = pca.transform(data3)
feature4 = pca.transform(data4)


# -------------------------
# Plot PCA
# -------------------------

plt.figure(figsize=(10, 7))

plt.plot(feature0[:, 0], feature0[:, 1], "or", label="N")
plt.plot(feature1[:, 0], feature1[:, 1], "ob", label="S")
plt.plot(feature2[:, 0], feature2[:, 1], "xg", label="F")
plt.plot(feature3[:, 0], feature3[:, 1], "vc", label="V")
plt.plot(feature4[:, 0], feature4[:, 1], "ok", label="U")

plt.legend()
plt.show()
