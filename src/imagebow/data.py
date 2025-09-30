import os, numpy as np
from typing import Tuple, List
from PIL import Image

# STL-10 classes
CLASSES = ["airplane","bird","car","cat","deer","dog","horse","monkey","ship","truck"]

def _list_images(folder: str) -> List[str]:
    exts = {".png",".jpg",".jpeg",".bmp"}
    paths = []
    for root, _, files in os.walk(folder):
        for n in files:
            if os.path.splitext(n)[1].lower() in exts:
                paths.append(os.path.join(root, n))
    paths.sort()
    return paths

def load_stl10_raw(root: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load STL-10 images from stl10_raw/ (train/test folders)."""
    base = os.path.join(root, split)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"{base} not found")
    X, y = [], []
    for ci, cname in enumerate(CLASSES):
        cdir = os.path.join(base, cname)
        if not os.path.isdir(cdir):
            continue
        for p in _list_images(cdir):
            im = Image.open(p).convert("RGB").resize((96,96))
            X.append(np.asarray(im, dtype=np.float32)/255.0)
            y.append(ci)
    return np.stack(X), np.array(y, dtype=np.int64)

def load_stl10_binary(root: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load STL-10 from binary .bin files."""
    x_path = os.path.join(root, f"{split}_X.bin")
    y_path = os.path.join(root, f"{split}_y.bin")
    X = np.fromfile(x_path, dtype=np.uint8).reshape(-1,3,96,96).transpose(0,2,3,1).astype(np.float32)/255.0
    y = np.fromfile(y_path, dtype=np.uint8).astype(np.int64) - 1  # labels are 1..10 → 0..9
    return X, y

def load_stl10(data_root: str, split: str="train") -> Tuple[np.ndarray, np.ndarray]:
    """Try raw first, then binary."""
    raw = os.path.join(data_root, "stl10_raw")
    if os.path.isdir(raw):
        return load_stl10_raw(raw, split)
    bin_root = os.path.join(data_root, "stl10_binary")
    if os.path.isdir(bin_root):
        return load_stl10_binary(bin_root, split)
    raise FileNotFoundError("Expected data_root/stl10_raw or data_root/stl10_binary")
# Returns (N,96,96,3) array of images and (N,) array of labels
# --- IGNORE ---



