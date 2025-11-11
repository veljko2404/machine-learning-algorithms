import numpy as np
import matplotlib.pyplot as plt

# d(xnew, xi) = sqrt(Σ(xnewj - xij)^2)

np.random.seed(0)
class0 = np.random.randn(50, 2) + np.array([-2, -2])
class1 = np.random.randn(50, 2) + np.array([2, 2])

X = np.vstack((class0, class1))
y = np.array([0]*50 + [1]*50)

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a -  b)**2))

def predict(X_train, y_train, x_new, k):
    distances = [euclidean_distance(x_new, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_neighbours = y_train[k_indices]
    values, counts = np.unique(k_neighbours, return_counts=True)
    return values[np.argmax(counts)]

def plot_decision_boundary(X, y, points, k=3):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    preds = np.array([predict(X, y, p, k) for p in grid_points])
    preds = preds.reshape(xx.shape)

    plt.contourf(xx, yy, preds, cmap="bwr", alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
    plt.scatter(points[:, 0], points[:, 1], c='green', s=100, marker='x')
    plt.title(f"KNN Decision Boundary k={k}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

points = np.array([[0, 0], [1, 0], [0, 1], [1, 2], [-1, 1], [-2, -1]])

k = 7

for point in points:
    print(predict(X, y, point, k))

plot_decision_boundary(X, y, points, k)
