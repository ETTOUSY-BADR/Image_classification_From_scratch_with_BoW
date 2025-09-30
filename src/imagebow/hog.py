import numpy as np
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.feature import hog

# -----------------------------
# Custom Dense HOG (128-D per patch)
# -----------------------------

def dense_grid_indices(w: int, h: int, nx: int, ny: int):
    xs = np.linspace(8, w-8, nx)
    ys = np.linspace(8, h-8, ny)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    return gx.ravel().astype(int), gy.ravel().astype(int)

def _hog_patch(gray: np.ndarray, cx: int, cy: int):
    patch = gray[cy-8:cy+8, cx-8:cx+8]
    gx = ndimage.sobel(patch, axis=1, mode="reflect")
    gy = ndimage.sobel(patch, axis=0, mode="reflect")
    mag = np.hypot(gx, gy) + 1e-6
    ang = (np.arctan2(gy, gx) + 2*np.pi) % (2*np.pi)

    hist = []
    for cyi in range(4):
        for cxi in range(4):
            cell = mag[cyi*4:(cyi+1)*4, cxi*4:(cxi+1)*4]
            a = ang[cyi*4:(cyi+1)*4, cxi*4:(cxi+1)*4]
            h, _ = np.histogram(a, bins=np.linspace(0,2*np.pi,9), weights=cell)
            hist.append(h.astype(np.float32))
    f = np.concatenate(hist).astype(np.float32)   # 16*8=128D
    return f / (np.linalg.norm(f) + 1e-6)

def dense_hog(img: np.ndarray, nx: int=6, ny: int=6) -> np.ndarray:
    """Return (nx*ny,128) HOG descriptors."""
    if img.ndim == 3:
        gray = (0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]).astype(np.float32)
    else:
        gray = img.astype(np.float32)
    h, w = gray.shape[:2]
    xs, ys = dense_grid_indices(w, h, nx, ny)
    return np.stack([_hog_patch(gray, x, y) for x, y in zip(xs, ys)], axis=0)

# -----------------------------
# Skimage HOG (baseline)
# -----------------------------

def extract_hog(img: np.ndarray, pixels_per_cell=(8,8), cells_per_block=(2,2)) -> np.ndarray:
    """Single feature vector using skimage HOG."""
    gray = rgb2gray(img)
    features = hog(
        gray,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True
    )
    return features
# img is (H,W,3) array; returns (N,128) array of descriptors
# --- IGNORE ---