import numpy as np
import matplotlib.pyplot as plt

# y = w1x1 + ... + wnxn + b - Linear Regression Model
# L = 1/m * Σ(yi - ŷi)^2 - Mean Squared Error
# w -= α * ∂L/∂w - Weight Update Rule
# b -= α * ∂L/∂b - Bias Update Rule

np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 3.5 * X + np.random.randn(100, 1)

w = np.random.randn(1, 1)
b = np.zeros((1, 1))
learning_rate = 0.001
epochs = 1000
m = len(X)

for epoch in range(epochs):
    y_pred = X.dot(w) + b
    error = y_pred - y

    dw = (2/m) * X.T.dot(error)
    dy = (2/m) * np.sum(error)
    w -= learning_rate * dw
    b -= learning_rate * dy

    if epoch % 100 == 0:
        loss = np.mean(error**2)
        print(f'Epoch {epoch}, Loss: {loss}')

y_pred_final = X.dot(w) + b
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, y_pred_final, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
print(f'Final weights: {w.flatten()}, Final bias: {b.flatten()}')