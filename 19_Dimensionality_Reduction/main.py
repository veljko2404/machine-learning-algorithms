import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# PCA – Principal Component Analysis

np.random.seed(42)
class1 = np.random.randn(100, 5) + np.array([0, 0, 0, 0, 0])
class2 = np.random.randn(100, 5) + np.array([5, 2, -1, 3, -2])
class3 = np.random.randn(100, 5) + np.array([-3, 4, 2, -4, 1])
class4 = np.random.randn(100, 5) + np.array([2, -3, 4, 1, 2])

X = np.vstack([class1, class2, class3, class4])
y = np.array([0]*100 + [1]*100 + [2]*100 + [3]*100)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total retained variance:", pca.explained_variance_ratio_.sum())

colors = ['blue', 'red', 'green', 'purple']
labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3']

plt.figure(figsize=(7,6))
for i in range(4):
    plt.scatter(
        X_pca[y==i,0],
        X_pca[y==i,1],
        color=colors[i],
        label=labels[i],
        alpha=0.7)

plt.title("PCA – Dimensionality Reduction to 2D")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True)
plt.show()

