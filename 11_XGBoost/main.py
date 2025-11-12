import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# yi = Σ(fk(xi))
# Not implemented from scratch

X, y = make_classification(n_samples=250, n_features=2, 
                           n_redundant=0, n_clusters_per_class=1,
                           n_informative=2, random_state=23)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = xgb.XGBClassifier(
    n_estimators = 100,
    learning_rate = 0.1,
    max_depth = 6,
    subsample = 0.8,
    colsample_bytree = 0.8,
    random_state = 0
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(acc)

x_min, x_max = X[:, 0].min() - 0.5, X[:,].max() + 0.5
y_min, y_max = X[:, 0].min() - 0.5, X[:,].max() + 0.5

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),np.linspace(y_min, y_max, 200))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap="bwr", alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
plt.title("XGBoost Decision Boundary")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()