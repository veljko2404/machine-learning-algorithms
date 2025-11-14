import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# Isolation Forest, One-Class SVM, LOF

np.random.seed(42)
normal = 0.3 * np.random.randn(300, 2)
anomalies = np.random.uniform(low=4, high=7, size=(12, 2))

X = np.vstack([normal, anomalies])

iso = IsolationForest(contamination=0.03, random_state=42)
iso_pred = iso.fit_predict(X)

ocsvm = OneClassSVM(kernel="rbf", gamma=0.1, nu=0.03)
ocsvm_pred = ocsvm.fit_predict(X)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.03)
lof_pred = lof.fit_predict(X)

models = {
    "Isolation Forest": iso_pred,
    "One-Class SVM": ocsvm_pred,
    "Local Outlier Factor": lof_pred
}

def plot_results(pred, title):
    plt.figure(figsize=(6,5))
    normal_points = X[pred == 1]
    anomaly_points = X[pred == -1]
    plt.scatter(normal_points[:,0], normal_points[:,1], c="blue", label="Normal", alpha=0.6)
    plt.scatter(anomaly_points[:,0], anomaly_points[:,1], c="red", label="Anomaly", s=80, edgecolor="k")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

for name, pred in models.items():
    plot_results(pred, name)
