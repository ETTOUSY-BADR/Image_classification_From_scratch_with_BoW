import numpy as np
from scipy.spatial.distance import cdist

def kmeans(X: np.ndarray, K: int, n_iter: int=10, seed: int=0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    C = X[rng.permutation(X.shape[0])[:K]].copy()
    for _ in range(n_iter):
        D = cdist(X, C, metric="euclidean")
        labels = np.argmin(D, axis=1)
        newC = np.zeros_like(C)
        for k in range(K):
            pts = X[labels==k]
            newC[k] = pts.mean(axis=0) if len(pts) else X[rng.randint(0, X.shape[0])]
        if np.allclose(newC, C): 
            C = newC; break
        C = newC
    return C
def quantize(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    D = cdist(X, C, metric="euclidean")
    return np.argmin(D, axis=1)
# X is (N,D), C is (K,D); returns (N,) array of cluster indices
# --- IGNORE ---



