import numpy as np
from imagebow.knn import predict_knn, accuracy

def test_knn_predict():
    Xtr = np.array([[0,0],[1,1]], dtype=np.float32); ytr = np.array([0,1])
    Xte = np.array([[0.1,0.1],[0.9,0.9]], dtype=np.float32)
    yp  = predict_knn(Xtr, ytr, Xte, metric="l2")
    assert accuracy(np.array([0,1]), yp) == 1.0
# Tests for predict_knn and accuracy functions
# --- IGNORE ---