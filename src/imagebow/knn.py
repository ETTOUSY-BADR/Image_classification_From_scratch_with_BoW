import numpy as np
from scipy.spatial.distance import cdist

def predict_knn(Xtr, ytr, Xte, metric="l2"):
    m = "euclidean" if metric=="l2" else "cityblock"
    D = cdist(Xte, Xtr, metric=m)
    return ytr[np.argmin(D, axis=1)]

def accuracy(y_true, y_pred) -> float:
    return float((y_true==y_pred).mean())
def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm
#def precision_recall(cm: np.ndarray) -> (np.ndarray, np.ndarray):
 #   tp = np.diag(cm).astype(np.float32)
  #  p = tp / (cm.sum(axis=0) + 1e-6)
   # r = tp / (cm.sum(axis=1) + 1e-6)
    # return p, r
# Xtr is (Ntr,D), ytr is (Ntr,), Xte is (Nte,D)
# Returns (Nte,) array of predicted labels
# --- IGNORE ---
