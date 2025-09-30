import numpy as np
from imagebow.hog import dense_hog

def test_dense_hog_shape():
    img = np.random.rand(96,96,3).astype(np.float32)
    H = dense_hog(img, nx=6, ny=6)
    assert H.shape == (36,128)
    assert np.isfinite(H).all()
    assert np.linalg.norm(H, axis=1).max() <= 1.0 + 1e-5
    assert np.linalg.norm(H, axis=1).min() >= 0.0 - 1e-5
# Tests for dense_hog function
# --- IGNORE ---

