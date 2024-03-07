from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plt_to_cv2(plt_figure):
    """Convert a matplotlib figure to a OpenCV image"""
    buf = BytesIO()
    plt_figure.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    return img


# Load data from binary file
with open(".optpath.bin", "rb") as fp:
    N = np.frombuffer(fp.read(4), dtype=np.int32)[0]
    D = np.frombuffer(fp.read(4), dtype=np.int32)[0]
    M = np.frombuffer(fp.read(4), dtype=np.int32)[0]
    L = np.frombuffer(fp.read(4), dtype=np.int32)[0]
    T = np.frombuffer(fp.read(D * M * L * 8), dtype=np.float64).reshape(D, M, L, order="F")
    X = np.frombuffer(fp.read(D * N * 8), dtype=np.float64).reshape(D, N, order="F")


sub = 1
# Re-initializing the title variables
title1 = "Source and target point sets"
title2 = "Optimization trajectory"

# Reshape and re-initialize variables
X = X.T
Y0 = T[:, :, 0].T
bbox = [min(X[:, 0]), max(X[:, 0]), min(X[:, 1]), max(X[:, 1])]
w0 = np.array([0, 0, 2, 1]) * 800
w1 = np.array([0, 0, 1, 1]) * 800
if sub == 1:
    w1 = w0
# Plotting only the final frame
plt.figure(figsize=(w1[2] / 100, w1[3] / 100))

if sub == 1:
    plt.subplot(1, 2, 1)
plt.plot(X[:, 0], X[:, 1], "bo", markersize=8)
plt.plot(Y0[:, 0], Y0[:, 1], "ro", markersize=5, markerfacecolor=[1, 0, 0])
plt.title(title1, fontsize=18)
plt.axis("equal")
plt.axis("off")

if sub == 1:
    plt.subplot(1, 2, 2)
Y_final = T[:, :, -1].T  # Using the last frame
plt.plot(X[:, 0], X[:, 1], "bo", markersize=8)
plt.plot(Y_final[:, 0], Y_final[:, 1], "ro", markersize=5, markerfacecolor=[1, 0, 0])
plt.title(title2, fontsize=18)
plt.axis(bbox)
plt.axis("equal")
plt.axis("off")

plt.show()
