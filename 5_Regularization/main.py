import numpy as np
import matplotlib.pyplot as plt

# L = 1/m * Σ(ŷi-yi)^2 + λ * Σ(w_j^2) - Ridge Regularization L2
# L = 1/m * Σ(ŷi-yi)^2 + λ * Σ|w_j| - Lasso Regularization L1
# L = MSE + λ1 * Σ|wj| + λ2 * Σ(wj^2) - Elastic Net Regularization

np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2.5 * X + 7 + np.random.randn(100, 1) * 2

w = np.random.randn(1, 1)
b = 0
learning_rate = 0.001
epochs = 2000
lam = 0.1
m, n = X.shape

for epoch in range(epochs):
    y_pred = X.dot(w) + b
    error = y_pred - y

    dw = (2/m) * (X.T.dot(error) + lam * w) # Ridge
    # dw = (2/m) * X.T.dot(error) + lam * np.sign(w) - Lasso
    # dw = (2/m) * X.T.dot(error) + lam1 * np.sign(w) + 2*lam2 * w - Elastic net
    db = (2/m) * np.sum(error)
    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 200 == 0:
        loss = np.mean(error**2) + lam * np.sum(w**2)
        print(f'Epoch {epoch}, Loss: {loss}')

plt.scatter(X, y, color="blue", label="Real data")
plt.plot(X, X.dot(w) + b, color="red", label="Ridge Regression Model")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Ridge Regularization in Linear Regression")
plt.legend()
plt.show()
