import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import numpy as np

# Reproducibility
np.random.seed(42)

# Create figure and styling for plotting
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
ax.set(xlabel='dimensions (m)', ylabel='log(dmax/dmin)',
       title='dmax/dmin vs. dimensionality')
line_styles = {0: 'ro-', 1: 'b^-', 2: 'gs-', 3: 'cv-'}

# We try a few sample sizes so we can see how the effect changes with N.
sample_sizes = [100, 500, 1000, 5000]

# Sweep dimensionality from 1 to 100.
feature_range = list(range(1, 101))

for idx, num_samples in enumerate(sample_sizes):
    ratios = []
    for num_features in feature_range:
        # Generate a Gaussian dataset. make_classification needs at least
        # n_informative + n_redundant + n_repeated <= n_features, so for very
        # low dimensionality we keep things minimal. We're not actually using
        # the labels here -- only the feature matrix X.
        n_informative = max(1, min(2, num_features))
        X, _ = make_classification(
            n_samples=num_samples,
            n_features=num_features,
            n_informative=n_informative,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=1,
            random_state=42,
        )

        # Pick a random query point from X
        q_idx = np.random.randint(0, len(X))
        query_point = X[q_idx]

        # Remove the query point so it isn't compared to itself (distance 0)
        X_rest = np.delete(X, q_idx, axis=0)

        # Euclidean distances from the query point to every other point
        distances = np.linalg.norm(X_rest - query_point, axis=1)

        ratio = np.max(distances) / np.min(distances)
        ratios.append(ratio)

    ax.plot(feature_range, np.log(ratios), line_styles[idx],
            label=f'N={num_samples:,}', markersize=3, linewidth=1)

plt.legend()
plt.tight_layout()
plt.grid(True)
plt.savefig('curse_of_dimensionality.png', dpi=120, bbox_inches='tight')
plt.show()
