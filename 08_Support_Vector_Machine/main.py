import numpy as np
import matplotlib.pyplot as plt

# yi(w^T * xi + b) >= 1
# L = 1/2 * ||w||^2 + C * Σ(max(0, 1 - yi * (w^T * xi + b)))

np.random.seed(0)
class0 = np.random.randn(50, 2) + np.array([-2, -2])
class1 = np.random.randn(50, 2) + np.array([2, 2])
X = np.vstack((class0, class1))
y = np.array([-1]*50 + [1]*50)

m, n = X.shape
w = np.zeros(n)
b = 0
C = 1.0
learning_rate = 0.001
epochs = 1000

for epoch in range(epochs):
    for i in range(m):
        condition = y[i] * (np.dot(X[i], w) + b)
        if condition >= 1:
            dw = w
            db = 0
        else:
            dw = w - C * y[i] * X[i]
            db = -C * y[i]
        w -= learning_rate * dw
        b -= learning_rate * db

x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z > 0, cmap="bwr", alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
plt.title("Linear SVM Decision Boundary")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

print("Final weights:", w)
print("Bias:", b)