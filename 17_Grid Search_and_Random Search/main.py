import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import uniform

np.random.seed(42)
class0 = np.random.randn(100, 2) + np.array([-2, -2])
class1 = np.random.randn(100, 2) + np.array([2, 2])
X = np.vstack([class0, class1])
y = np.array([0]*100 + [1]*100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
param_grid = {
    "C": [0.1, 1, 10, 50, 100],
    "gamma": [0.01, 0.1, 1],
    "kernel": ["rbf"]
}

grid = GridSearchCV(SVC(), param_grid, cv=5, verbose=0)
grid.fit(X_train, y_train)

print("GRID SEARCH BEST PARAMS:", grid.best_params_)
print("GRID SEARCH BEST SCORE:", grid.best_score_)

best_grid_model = grid.best_estimator_
y_pred_grid = best_grid_model.predict(X_test)
print("GRID SEARCH TEST ACC:", accuracy_score(y_test, y_pred_grid))

param_dist = {
    "C": uniform(0.1, 100),
    "gamma": uniform(0.001, 1),
    "kernel": ["rbf"]
}

random_search = RandomizedSearchCV(
    SVC(),
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    random_state=42,
    verbose=0
)

random_search.fit(X_train, y_train)

print("\nRANDOM SEARCH BEST PARAMS:", random_search.best_params_)
print("RANDOM SEARCH BEST SCORE:", random_search.best_score_)

best_random_model = random_search.best_estimator_
y_pred_random = best_random_model.predict(X_test)
print("RANDOM SEARCH TEST ACC:", accuracy_score(y_test, y_pred_random))

def plot_decision_boundary(model, title):
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, Z, alpha=0.35, cmap='coolwarm')
    plt.scatter(class0[:,0], class0[:,1], c='blue', label='Class 0')
    plt.scatter(class1[:,0], class1[:,1], c='red', label='Class 1')
    plt.title(title)
    plt.legend()
    plt.show()

plot_decision_boundary(best_grid_model, "Decision Boundary (Grid Search)")
plot_decision_boundary(best_random_model, "Decision Boundary (Random Search)")
