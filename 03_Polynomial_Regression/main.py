import numpy as np
import matplotlib.pyplot as plt

# y = w1x + w2x^2 + ... + wnx^n + b
# L = 1/m * Σ(y_pred - y_true)^2
# L = 1/m * (Xpoly*w + b - y)^T * (Xpoly*w + b - y)
# dL/dw = 2/m * Xpoly^T * (Xpoly*w + b - y)
# dL/db = 2/m * Σ(Xpoly*w + b - y)
# w = w - α * dL/dw
# b = b - α * dL/db

np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 0.5 * X**3 - 2 * X**2 + X + 3 + np.random.randn(100, 1) * 3

degree = 3
X_poly = np.hstack([X**i for i in range(1, degree + 1)])
m, n = X_poly.shape
w = np.random.randn(n, 1)
b = np.zeros((1, 1))
learning_rate = 0.001
epochs = 1000

for epoch in range(epochs):
    y_pred = X_poly.dot(w) + b
    error = y_pred - y

    dw = (2/m) * X_poly.T.dot(error)
    db = (2/m) * np.sum(error)

    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 100 == 0:
        loss = np.mean(error**2)
        print(f'Epoch {epoch}, Loss: {loss}')

y_pred_final = X_poly.dot(w) + b

plt.scatter(X, y, color="blue", label="Real data")
plt.plot(X, y_pred_final, color="red", label="Polynomial Regression Fit")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Polynomial Regression Visualization")
plt.legend()
plt.show()
print("Trained weights:", w.flatten())
print("Trained bias:", b[0][0])