# Image Classification from Scratch (BoW + HOG + kNN)

# Bag-of-Words Classification with HOG (Histograms of Oriented Gradients)

##  Introduction
The **Histogram of Oriented Gradients (HOG)** is a powerful descriptor used in computer vision and OCR.  
It captures **shape and contour information** of objects by analyzing local gradient orientations.  
HOG-based features are widely used in **text and digit recognition**, since characters are defined by strokes and edges.

---

##  HOG Feature Extraction
1. **Gradient computation**  
   - Compute gradients (e.g., with Sobel operators).  
   - Each pixel has a gradient vector (magnitude + direction).

2. **Orientation binning**  
   - Gradients are quantized into **4 or 8 directions**.  
   - Votes to histogram bins are weighted by gradient magnitude.

3. **Local cell histograms**  
   - Divide the image into small **cells** (e.g., 8×8 pixels).  
   - Accumulate histogram bins inside each cell.

4. **Block normalization**  
   - Normalize histograms across larger **blocks** (e.g., 16×16).  
   - Reduces sensitivity to **illumination changes**.

---

## Bag-of-Words (BoW) with HOG
The **Bag-of-Words (BoW)** framework converts local descriptors into a global representation for classification.

1. **Feature extraction** → Extract HOG descriptors from patches in training images.  
2. **Codebook generation** → Use **k-means clustering** to create a *visual vocabulary*.  
3. **Histogram representation** → Assign each HOG descriptor to its nearest cluster center.  
4. **Classification** → Train a classifier (e.g., **SVM, Logistic Regression, Random Forest**) on the histogram vectors.

---

##  Advantages
- Captures **local structure** (strokes, edges, shapes).  
- **Illumination invariant** thanks to block normalization.  
- Robust to **deformations and noise**.  
- Effective for **OCR tasks**, including handwritten or printed digits.

---

## Workflow Diagram
```plaintext
Input Image 
   ↓
Gradient Computation 
   ↓
HOG Descriptors 
   ↓
K-Means Codebook 
   ↓
Histogram (BoW) 
   ↓
Classifier


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

