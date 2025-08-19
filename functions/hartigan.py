from typing import Iterable, Hashable

import numpy as np


def _validate_dist_and_labels(labels: Iterable[Hashable], dist_matrix: np.ndarray):
    labels = np.asarray(list(labels))
    D = np.asarray(dist_matrix)

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("dist_matrix must be a square (n x n) array.")
    n = D.shape[0]
    if labels.shape[0] != n:
        raise ValueError("labels length must match dist_matrix size.")
    if np.any(D < 0):
        raise ValueError("dist_matrix must be non-negative.")
    if not np.allclose(np.diag(D), 0):
        raise ValueError("dist_matrix diagonal must be all zeros.")
    if not np.allclose(D, D.T, atol=1e-8):
        raise ValueError("dist_matrix must be symmetric.")
    return labels, D

def hartigan_wcss(
    labels: Iterable[Hashable],
    dist_matrix: np.ndarray,
    *,
    squared: bool = False,
    ignore_noise: bool = False,
    noise_label: Hashable = -1,
) -> float:
    """
    Hartigan's total within-cluster sum of squares W_k computed from a distance matrix.

    For a cluster C with size |C|, using Euclidean distances:
        W(C) = (1 / (2*|C|)) * sum_{i,j in C} d(i,j)^2
    and W_k = sum over clusters C of W(C).

    Parameters
    ----------
    labels : iterable
        Cluster labels for each sample (same order as dist_matrix).
    dist_matrix : (n x n) ndarray
        Symmetric, non-negative distances with zeros on the diagonal.
    squared : bool, default False
        If True, `dist_matrix` entries are already squared distances (d^2).
        If False, they will be squared internally.
    ignore_noise : bool, default False
        If True, drop all points whose label == noise_label (useful for DBSCAN).
    noise_label : hashable, default -1
        Label treated as noise when ignore_noise=True.

    Returns
    -------
    float
        Total within-cluster sum of squares W_k.
        Returns np.nan if there are fewer than 1 non-noise points or no clusters.
    """
    labels, D = _validate_dist_and_labels(labels, dist_matrix)

    # Optionally filter noise
    if ignore_noise:
        mask = labels != noise_label
        labels = labels[mask]
        D = D[np.ix_(mask, mask)]
        if labels.size == 0:
            return np.nan

    D2 = D if squared else D ** 2

    # Map clusters -> indices
    clusters = {}
    for idx, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(idx)

    if len(clusters) == 0:
        return np.nan

    W = 0.0
    for inds in clusters.values():
        m = len(inds)
        if m <= 1:
            continue  # singleton contributes 0
        sub = D2[np.ix_(inds, inds)]
        # includes diagonal zeros; that's fine
        W += float(sub.sum()) / (2.0 * m)
    return W

def hartigan_index(
    labels_k: Iterable[Hashable],
    labels_kplus1: Iterable[Hashable],
    dist_matrix: np.ndarray,
    *,
    squared: bool = False,
    ignore_noise: bool = False,
    noise_label: Hashable = -1,
) -> float:
    """
    Compute Hartigan's index:
        H_k = (W_k / W_{k+1} - 1) * (n - k - 1)

    Notes
    -----
    - This classic form is used to pick the number of clusters with k-means.
    - `labels_k` should reflect a partition into k clusters, and `labels_kplus1`
      a partition of the *same* points into k+1 clusters (ideally their respective
      best solutions).
    - W_k is computed from the (squared) distance matrix using the
      pairwise formula, which matches Euclidean WCSS exactly if distances are Euclidean.

    Returns
    -------
    float
        Hartigan's index H_k. Larger values typically indicate that k clusters
        are insufficient (rule-of-thumb: values > ~10 suggest increasing k).
        Returns np.nan if undefined (e.g., W_{k+1} == 0 with W_k == 0, or <2 effective clusters).
    """
    # Validate dimensions once (also checks symmetry / sizes)
    labels_all, D = _validate_dist_and_labels(labels_k, dist_matrix)

    # Build a mask if ignoring noise to ensure both labelings drop the same points
    if ignore_noise:
        mask = (np.asarray(list(labels_k)) != noise_label) & (np.asarray(list(labels_kplus1)) != noise_label)
        if not np.any(mask):
            return np.nan
        D = D[np.ix_(mask, mask)]
        labels_k = np.asarray(list(labels_k))[mask]
        labels_kplus1 = np.asarray(list(labels_kplus1))[mask]
    else:
        labels_k = np.asarray(list(labels_k))
        labels_kplus1 = np.asarray(list(labels_kplus1))

    # n and k as used in the formula
    n = D.shape[0]
    k = len(np.unique(labels_k))

    if n <= k + 1:
        return np.nan  # (n - k - 1) would be <= 0; index not meaningful

    Wk  = hartigan_wcss(labels_k,        D, squared=squared)
    Wk1 = hartigan_wcss(labels_kplus1,   D, squared=squared)

    if not np.isfinite(Wk) or not np.isfinite(Wk1):
        return np.nan
    if Wk1 == 0.0:
        if Wk == 0.0:
            return np.nan  # degenerate (all points coincide / all singletons)
        return np.inf

    return (Wk / Wk1 - 1.0) * (n - k - 1.0)
