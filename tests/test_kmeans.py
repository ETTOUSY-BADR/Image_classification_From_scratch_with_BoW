import numpy as np
from imagebow.kmeans import kmeans

def test_kmeans_basic():
    X = np.vstack([np.zeros((10,2)), np.ones((10,2))]).astype(np.float32)
    C = kmeans(X, K=2, n_iter=5, seed=0)
    assert C.shape == (2,2)
    assert np.linalg.norm(C[0]-C[1]) > 0.5
# Basic test for kmeans function
# --- IGNORE ---
