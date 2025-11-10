import numpy as np
import matplotlib.pyplot as plt

# y = w1x1 + ... + wnxn + b - Multiple Linear Regression Model
# L = 1/m * Σ(yi - ŷi)^2 - Mean Squared Error
# w -= α * ∂L/∂w - Weight Update Rule
# b -= α * ∂L/∂b - Bias Update Rule

np.random.seed(42)
X1 = np.random.rand(100, 1) * 10
X2 = np.random.rand(100, 1) * 5
y = 5 * X1 + 3 * X2 + 10 + np.random.randn(100, 1) * 2

X = np.hstack((X1, X2))
n_features = X.shape[1]
w = np.random.randn(n_features, 1)
b = np.zeros((1, 1))
learning_rate = 0.0005
epochs = 500
m = len(X)

for epoch in range(epochs):
    y_pred = X.dot(w) + b
    error = y_pred - y

    dw = (2/m) * X.T.dot(error)
    db = (2/m) * np.sum(error)

    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 50 == 0:
        loss = np.mean(error**2)
        print(f'Epoch {epoch}, Loss: {loss}')

y_pred_final = X.dot(w) + b

X1_range = np.linspace(X1.min(), X1.max(), 100).reshape(-1, 1)
X2_mean = np.full_like(X1_range, X2.mean())
X_plot = np.hstack((X1_range, X2_mean))
y_plot = X_plot.dot(w) + b

plt.scatter(X1, y, color="blue", label="Real data")
plt.plot(X1_range, y_plot, color="red", label="Model fit (X2 fixed at mean)")
plt.xlabel("X1")
plt.ylabel("y")
plt.title("Multiple Linear Regression Visualization")
plt.legend()
plt.show()