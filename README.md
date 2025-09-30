# Image Classification from Scratch (BoW + HOG + kNN)

This project implements an **image classification pipeline** from scratch:
HOG descriptors → K-means codebook → Bag-of-Words histograms → 1-NN classifier.

## Dataset
We use **STL-10**:
- 10 classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck
- Train: 500 images per class (5,000 total)
- Test: 800 images per class (8,000 total)
- Size: 96×96 RGB

Put your data inside `data/stl10_raw/train/...` and `data/stl10_raw/test/...`.

## Quickstart
```bash
# Create environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Two classes (airplane vs car)
python -m experiments.two_classes

# All 10 classes
python -m experiments.all_classes
