import os, sys, json, numpy as np, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from imagebow.data import load_stl10, CLASSES
from imagebow.hog import dense_hog
from imagebow.kmeans import kmeans
from imagebow.bow import compute_bow
from imagebow.knn import predict_knn, accuracy

def subset_two(X, y, a, b):
    ia, ib = CLASSES.index(a), CLASSES.index(b)
    m = (y==ia) | (y==ib)
    return X[m], (y[m]==ib).astype(np.int64)

def sample_descs(X, n_per=50, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for img in X:
        H = dense_hog(img)
        idx = rng.choice(H.shape[0], size=min(n_per, H.shape[0]), replace=False)
        out.append(H[idx])
    return np.vstack(out)

def extract_lists(X): return [dense_hog(img) for img in X]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data_STL10")
    ap.add_argument("--a", type=str, default="airplane")
    ap.add_argument("--b", type=str, default="car")
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--metric", type=str, default="l2", choices=["l1","l2"])
    ap.add_argument("--norm", type=str, default="l1", choices=["l1","l2"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Xtr,ytr = load_stl10(args.data_root, "train")
    Xte,yte = load_stl10(args.data_root, "test")
    Xtr,ytr = subset_two(Xtr,ytr,args.a,args.b)
    Xte,yte = subset_two(Xte,yte,args.a,args.b)

    rng = np.random.RandomState(args.seed)
    accs = []
    for r in range(args.runs):
        seed = int(rng.randint(0, 1_000_000))
        C = kmeans(sample_descs(Xtr, seed=seed), K=args.K, n_iter=args.iters, seed=seed)
        Htr = compute_bow(extract_lists(Xtr), C, args.norm)
        Hte = compute_bow(extract_lists(Xte), C, args.norm)
        yp  = predict_knn(Htr, ytr, Hte, args.metric)
        acc = accuracy(yte, yp); accs.append(acc)
        print(f"[Run {r+1}/{args.runs}] Acc = {acc:.4f}")

    print("Mean ± Std:", float(np.mean(accs)), float(np.std(accs)))

if __name__ == "__main__":
    main()
# Usage example:
# python experiments/two_classes.py --data_root data_STL10 --a airplane --b car --K 100 --iters 10 --runs 10 --metric l2 --norm l1 --seed 0
# Compares two STL-10 classes using BoW with HOG features and k-means clustering
# Prints accuracy for each run and mean ± std over all runs
# Example output:
# [Run 1/10] Acc = 0.8700
# Mean ± Std: 0.8765 0.0123
# --- IGNORE ---