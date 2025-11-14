import numpy as np
import matplotlib.pyplot as plt

# entropy = - Σ(pi * log2(pi))
# gini = 1 - Σ(pi^2)
# gain = impurity(parent) - weighted sum of impurity(childrem)

X = np.array([[2.7, 2.5],
              [1.3, 3.5],
              [3.0, 4.0],
              [3.2, 1.0],
              [2.0, 2.7],
              [1.0, 1.0]])
y = np.array([0, 0, 1, 1, 0, 0])

def gini(y):
    classes = np.unique(y)
    impurity = 1.0
    for c in classes:
        p = np.sum(y == c) / len(y)
        impurity -= p ** 2
    return impurity

def split(X, y, feature, threshold):
    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold
    return X[left_idx], X[right_idx], y[left_idx], y[right_idx]

def best_split(X, y):
    best_gain = 0
    best_feat, best_thresh = None, None
    current_impurity = gini(y)

    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for t in thresholds:
            _, _, y_left, y_right = split(X, y, feature, t)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            p = len(y_left) / len(y)
            gain = current_impurity - (p*gini(y_left) + (1-p)*gini(y_right))
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, feature, t
    return best_feat, best_thresh, best_gain

feat, thresh, gain = best_split(X, y)
print(f"NBest: feature {feat}, threshold {thresh}, gain {gain:.3f}")
