"""Simple DBSCAN clustering example using scikit-learn.

The script generates a two-moon dataset, standardizes the features,
runs DBSCAN, and visualizes the resulting clusters.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler


def main() -> None:
    """Generate sample data, fit DBSCAN, and plot cluster assignments."""

    # Create a noisy two-moon dataset that DBSCAN can recover.
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)

    # Standardize features so DBSCAN's distance-based threshold works well.
    X_scaled = StandardScaler().fit_transform(X)

    # Fit the clustering model; tweak eps/min_samples to explore behavior.
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    dbscan.fit(X_scaled)

    labels = dbscan.labels_

    # Identify core points for nicer plotting (optional but informative).
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True

    unique_labels = sorted(set(labels))
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        # Noise is labeled as -1 in scikit-learn's DBSCAN implementation.
        if label == -1:
            color = np.array([0.0, 0.0, 0.0, 1.0])

        class_member_mask = labels == label

        # Plot core samples.
        xy_core = X_scaled[class_member_mask & core_samples_mask]
        plt.plot(
            xy_core[:, 0],
            xy_core[:, 1],
            "o",
            markerfacecolor=tuple(color),
            markeredgecolor="k",
            markersize=8,
        )

        # Plot border samples.
        xy_border = X_scaled[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy_border[:, 0],
            xy_border[:, 1],
            "o",
            markerfacecolor=tuple(color),
            markeredgecolor="k",
            markersize=4,
        )

    plt.title("DBSCAN clustering on two moons")
    plt.xlabel("Feature 1 (standardized)")
    plt.ylabel("Feature 2 (standardized)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

