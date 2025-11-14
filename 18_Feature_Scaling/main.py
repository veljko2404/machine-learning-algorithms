import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# x' = (x - μ) / σ

np.random.seed(42)

x1 = np.random.randn(200) * 2 + 5
x2 = np.random.randn(200) * 500 + 2000
X = np.column_stack([x1, x2])

std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X)

mm_scaler = MinMaxScaler()
X_mm = mm_scaler.fit_transform(X)

plt.figure(figsize=(15,4))
plt.subplot(1,3,1)
plt.scatter(X[:,0], X[:,1], color='purple')
plt.title("Original data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.subplot(1,3,2)
plt.scatter(X_std[:,0], X_std[:,1], color='blue')
plt.title("StandardScaler (μ=0, σ=1)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.subplot(1,3,3)
plt.scatter(X_mm[:,0], X_mm[:,1], color='red')
plt.title("MinMaxScaler (0–1)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()
