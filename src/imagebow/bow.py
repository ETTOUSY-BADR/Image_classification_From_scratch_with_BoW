import numpy as np
from scipy.spatial.distance import cdist

def compute_bow(desc_lists, codebook: np.ndarray, norm: str="l1"):
    K = codebook.shape[0]
    H = np.zeros((len(desc_lists), K), dtype=np.float32)
    for i, D in enumerate(desc_lists):
        d = cdist(D, codebook, metric="euclidean")
        words = np.argmin(d, axis=1)
        hist = np.bincount(words, minlength=K).astype(np.float32)
        if norm=="l1":
            hist /= (hist.sum() + 1e-6)
        elif norm=="l2":
            hist /= (np.linalg.norm(hist) + 1e-6)
        H[i] = hist
    return H
# desc_lists is list of (Ni,D) arrays of descriptors
# codebook is (K,D) array of cluster centers
# Returns (len(desc_lists), K) array of histograms
# --- IGNORE ---


