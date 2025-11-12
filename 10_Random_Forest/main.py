import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
np.random.seed(0)
X = np.random.randn(300, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

def gini(y):
    classes = np.unique(y)
    impurity = 1.0
    for c in classes:
        p = np.sum(y == c) / len(y)
        impurity -= p ** 2
    return impurity

def split(X, y, feature, threshold):
    left = X[:, feature] <= threshold
    right = X[:, feature] > threshold
    return X[left], X[right], y[left], y[right]

def best_split(X, y, n_features):
    best_gain = -1
    best_feat, best_thresh = None, None
    features = np.random.choice(X.shape[1], n_features, replace=False)
    current_impurity = gini(y)
    
    for feat in features:
        thresholds = np.unique(X[:, feat])
        for t in thresholds:
            _, _, y_left, y_right = split(X, y, feat, t)
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            p = len(y_left) / len(y)
            gain = current_impurity - (p * gini(y_left) + (1 - p) * gini(y_right))
            if gain > best_gain:
                best_gain, best_feat, best_thresh = gain, feat, t
    return best_feat, best_thresh

def build_tree(X, y, depth=0, max_depth=5, n_features=None):
    if len(np.unique(y)) == 1 or depth >= max_depth:
        return Counter(y).most_common(1)[0][0]
    
    feat, thresh = best_split(X, y, n_features)
    if feat is None:
        return Counter(y).most_common(1)[0][0]
    
    X_left, X_right, y_left, y_right = split(X, y, feat, thresh)
    left_branch = build_tree(X_left, y_left, depth+1, max_depth, n_features)
    right_branch = build_tree(X_right, y_right, depth+1, max_depth, n_features)
    
    return (feat, thresh, left_branch, right_branch)

def predict_tree(x, tree):
    if not isinstance(tree, tuple):
        return tree
    feat, thresh, left, right = tree
    branch = left if x[feat] <= thresh else right
    return predict_tree(x, branch)

class RandomForest:
    def __init__(self, n_trees=5, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        for _ in range(self.n_trees):
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[idxs], y[idxs]
            tree = build_tree(X_sample, y_sample, max_depth=self.max_depth, n_features=int(np.sqrt(n_features)))
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([[predict_tree(x, tree) for x in X] for tree in self.trees])
        y_pred = [Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]
        return np.array(y_pred)

forest = RandomForest(n_trees=29, max_depth=4)
forest.fit(X, y)
preds = forest.predict(X)

accuracy = np.mean(preds == y)
print(f"Accuracy: {accuracy*100:.2f}%")

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
preds = forest.predict(grid)
Z = preds.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap="bwr", alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
plt.title("Random Forest Decision Boundary")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()