import numpy as np
import matplotlib.pyplot as plt

# z = w1x1 + ... + wnxn + b
# ŷ = 1 / (1 + e^(-z)) - Sigmoid Function
# L = -1/m * Σ[yi*log(ŷi) + (1-yi)*log(1-ŷi)] - Binary Cross-Entropy Loss

np.random.seed(0)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

m, n = X.shape
w = np.zeros((n, 1))
b = 0
learning_rate = 0.1
epochs = 1000

for epoch in range(epochs):
    z = X.dot(w) + b
    y_pred = sigmoid(z)
    loss = -(1/m) * np.sum(y*np.log(y_pred + 1e-9) + (1-y)*np.log(1-y_pred + 1e-9))

    dw = (1/m) * X.T.dot(y_pred - y)
    db = (1/m) * np.sum(y_pred - y)

    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = sigmoid(grid.dot(w) + b).reshape(xx1.shape)
print(probs)
plt.contourf(xx1, xx2, probs, cmap="RdBu", alpha=0.7)
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolors='k', cmap="bwr")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
print("Trained weights:", w.flatten())
print("Trained bias:", b)
