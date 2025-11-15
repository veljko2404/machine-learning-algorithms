import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

np.random.seed(22)

mean1 = [-3, -2]
cov1 = [[1.2, 0.6],
        [0.6, 1.0]]

mean2 = [3, 3]
cov2 = [[1.0, -0.4],
        [-0.4, 1.3]]

mean3 = [0, 6]
cov3 = [[1.5, 0.3],
        [0.3, 0.7]]

X1 = np.random.multivariate_normal(mean1, cov1, 200)
X2 = np.random.multivariate_normal(mean2, cov2, 200)
X3 = np.random.multivariate_normal(mean3, cov3, 200)
X = np.vstack([X1, X2, X3])

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

labels = gmm.predict(X)

plt.figure(figsize=(7,6))
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', alpha=0.7)
plt.title("Gaussian Mixture Models (GMM)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
