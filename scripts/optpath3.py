from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load data from binary file
with open(".optpath.bin", "rb") as fp:
    N = np.frombuffer(fp.read(4), dtype=np.int32)[0]
    D = np.frombuffer(fp.read(4), dtype=np.int32)[0]
    M = np.frombuffer(fp.read(4), dtype=np.int32)[0]
    L = np.frombuffer(fp.read(4), dtype=np.int32)[0]
    T = np.frombuffer(fp.read(D * M * L * 8), dtype=np.float64).reshape(D, M, L, order="F")
    X = np.frombuffer(fp.read(D * N * 8), dtype=np.float64).reshape(D, N, order="F")


def plt_to_cv2(plt_figure):
    """Convert a matplotlib figure to a OpenCV image"""
    buf = BytesIO()
    plt_figure.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    return img


# Define plotting functions
def plot_2d_views(X, T, png=0, sub=1, traj=1):
    title1 = "Before Registration"
    title2 = "Optimization Trajectory"

    if traj:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        lines = []

        for d in range(D):
            if d == 0:
                R = np.eye(3)
                idx = [0, 1, 2]
            else:
                a = 1 / np.sqrt(2)
                R = np.array([[a, -a, 0], [a, a, 0], [0, 0, 1]])
                idx = [2, 0, 1] if d == 1 else [0, 2, 1]

            Z = np.dot(R[idx][:, idx], X)
            X0 = Z.T

            for i in range(L):
                U = np.dot(R[idx][:, idx], T[:, :, i])
                Y = U.T

                if i == 0:
                    axs[0].plot(X0[:, 0], X0[:, 1], "b.", markersize=3)
                    line1, = axs[0].plot(Y[:, 0], Y[:, 1], "r.", markersize=3)
                    axs[0].set_aspect("equal", adjustable="box")
                    axs[0].axis("equal")
                    axs[0].axis("off")
                    axs[0].set_title(f"{title1}: View {d+1}", fontsize=18)
                    lines.append(line1)

                line2, = axs[1].plot(X0[:, 0], X0[:, 1], "b.", markersize=3)
                line3, = axs[1].plot(Y[:, 0], Y[:, 1], "r.", markersize=3)
                axs[1].set_aspect("equal", adjustable="box")
                axs[1].axis("equal")
                axs[1].axis("off")
                axs[1].set_title(f"{title2}: View {d+1} Iteration {i+1}", fontsize=18)
                lines.extend([line2, line3])

                # Save the plot as a PNG image
                if png == 1 and (i == L - 1 or i % 10 == 0):  # Example: Save at the last iteration or every 10 iterations
                    plt.savefig(f"Work/otw-v{d}-{i:04d}.png")

                # Convert the plot to an OpenCV image and display it
                if i == L - 1 or i % 1 == 0:  # Example: Display at the last iteration or every 10 iterations
                    img = plt_to_cv2(fig)
                    cv2.imshow(f"2D View {d+1}", img)
                    cv2.waitKey(1)

                # Remove the previous lines for the next iteration
                for line in lines:
                    line.remove()
                lines.clear()

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
