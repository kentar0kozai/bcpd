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
with open("../win/VisualStudio/BCPD-Win/.optpath.bin", "rb") as fp:
    N = np.frombuffer(fp.read(4), dtype=np.int32)[0]
    D = np.frombuffer(fp.read(4), dtype=np.int32)[0]
    M = np.frombuffer(fp.read(4), dtype=np.int32)[0]
    L = np.frombuffer(fp.read(4), dtype=np.int32)[0]
    T = np.frombuffer(fp.read(D * M * L * 8), dtype=np.float64).reshape(
        D, M, L, order="F"
    )
    X = np.frombuffer(fp.read(D * N * 8), dtype=np.float64).reshape(D, N, order="F")


# Define plotting functions
def plot_2d_views(X, T, png=1, sub=1, traj=1):
    title1 = "Before Registration"
    title2 = "Optimization Trajectory"

    if traj:
        for d in range(D):
            if d == 0:
                R = np.eye(3)
                idx = [0, 1, 2]
            else:
                a = 1 / np.sqrt(2)
                R = np.array([[a, -a, 0], [a, a, 0], [0, 0, 1]])
                idx = [2, 0, 1] if d == 1 else [0, 2, 1]

            Z = np.dot(R[idx][:, idx], X)

            # Draw the "Before Registration" subplot only once

            U = np.dot(R[idx][:, idx], T[:, :, 0])
            t1 = f"{title1}: View {d+1}"
            X0 = Z.T
            Y0 = U.T

            fig = plt.figure(figsize=(10, 5))
            plt.suptitle(f"View {d+1}")

            if sub == 1:
                plt.subplot(1, 2, 1)
                plt.plot(X0[:, 0], X0[:, 1], "b.", markersize=3)
                plt.plot(Y0[:, 0], Y0[:, 1], "r.", markersize=3)
                plt.gca().set_aspect("equal", adjustable="box")
                plt.axis("equal")
                plt.axis("off")
                plt.title(t1, fontsize=18)

            for l in range(L):
                U = np.dot(R[idx][:, idx], T[:, :, l])
                t2 = f"{title2}: View {d+1}"

                if sub == 1:
                    plt.subplot(1, 2, 2)
                Y = U.T
                W = Z.T
                plt.plot(W[:, 0], W[:, 1], "b.", markersize=3)
                plt.plot(Y[:, 0], Y[:, 1], "r.", markersize=3)
                plt.gca().set_aspect("equal", adjustable="box")
                plt.axis("equal")
                plt.axis("off")
                plt.title(t2, fontsize=18)

                if png == 1:
                    plt.savefig(f"Work/otw-v{d}-{l:04d}.png")

                img = plt_to_cv2(fig)
                cv2.imshow("2D" + str(d), img)
                cv2.waitKey(1)
                if sub == 1:
                    plt.subplot(1, 2, 2).clear()
                else:
                    plt.clf()
            plt.close()


def plot_3d_views(X, T):
    X0 = X.T
    T0 = T[:, :, 0].T
    T1 = T[:, :, -1].T

    fig = plt.figure(figsize=(10, 5))
    fig.suptitle("Before/After Registration")

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(T0[:, 0], T0[:, 1], T0[:, 2], c="r", marker=".", s=3)
    ax.scatter(X0[:, 0], X0[:, 1], X0[:, 2], c="b", marker=".", s=3)
    ax.set_title("Before Registration", fontsize=18)
    plt.axis("equal")

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.scatter(T1[:, 0], T1[:, 1], T1[:, 2], c="r", marker=".", s=3)
    ax.scatter(X0[:, 0], X0[:, 1], X0[:, 2], c="b", marker=".", s=3)
    ax.set_title("After Registration", fontsize=18)
    plt.axis("equal")
    plt.show()


# Execute plotting functions
plot_2d_views(X, T)
plot_3d_views(X, T)
