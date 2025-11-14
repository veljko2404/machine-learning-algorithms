import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(42)
N = 200

class0 = np.random.randn(100, 2) + np.array([-2, -2])
class1 = np.random.randn(100, 2) + np.array([2, 2])

X = np.vstack([class0, class1])
y = np.array([0]*100 + [1]*100)

model = LogisticRegression()

kf = KFold(n_splits=5, shuffle=True, random_state=0)

fold_acc = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    fold_acc.append(acc)

print("Accuracy by folds:", fold_acc)
print("Avg Accuracy:", np.mean(fold_acc))

model.fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(7,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(class0[:,0], class0[:,1], c='blue', label='Class 0')
plt.scatter(class1[:,0], class1[:,1], c='red', label='Class 1')
plt.title("Decision Boundary + Original Data")
plt.legend()
plt.show()
