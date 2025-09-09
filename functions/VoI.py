import numpy as np
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import mutual_info_score

def _entropy_from_probs(p, base=np.e):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]                 # ignore zeros
    if base == np.e:
        return -np.sum(p * np.log(p))
    else:
        return -np.sum(p * (np.log(p) / np.log(base)))

def variation_of_information(labels_a, labels_b, base=2):
    """
    Variation of Information between two clusterings.
    VI(X,Y) = H(X) + H(Y) - 2 I(X;Y)
    Lower is more similar. 0 means identical clusterings.

    Parameters
    ----------
    labels_a, labels_b : array-like of shape (n_samples,)
        Cluster labels (arbitrary integers/strings are fine).
    base : float, optional (default=2)
        Logarithm base for entropy/MI (2 -> bits, e -> nats).

    Returns
    -------
    vi : float
    """
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    if labels_a.shape[0] != labels_b.shape[0]:
        raise ValueError("labels_a and labels_b must have the same length")

    # Contingency (confusion) table
    C = contingency_matrix(labels_a, labels_b, sparse=False)
    n = C.sum()

    # Convert to joint / marginal distributions
    Pxy = C / n
    Px = Pxy.sum(axis=1)
    Py = Pxy.sum(axis=0)

    # Entropies and mutual information (use contingency directly for MI)
    Hx = _entropy_from_probs(Px, base=base)
    Hy = _entropy_from_probs(Py, base=base)
    Ixy = mutual_info_score(None, None, contingency=C) / (np.log(base))

    return Hx + Hy - 2.0 * Ixy