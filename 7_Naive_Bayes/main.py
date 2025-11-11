import numpy as np
import matplotlib.pyplot as plt

# P(A|B) = (P(B|A) * P(B)) / P(A)

np.random.seed(0)
class0 = np.random.randn(50, 2) + np.asarray([-2, -2])
class1 = np.random.randn(50, 2) + np.asarray([2, 2])

X = np.vstack((class0, class1))
y = np.array([0]*50 + [1]*50)

def fit(X, y):
    classes = np.unique(y)
    params = {}
    for c in classes:
        X_c = X[y == c]
        params[c] = {
            "mean": X_c.mean(axis=0),
            "var": X_c.var(axis=0)
        }
    priors = {c: len(y[y == c]) / len(y) for c in classes}
    return params, priors

def gaussian_prob(x, mean, var):
    eps = 1e-6
    coef = 1.0 / np.sqrt(2.0 * np.pi * var)
    exponent = np.exp(- (x - mean)**2 / (2 * var + eps))
    return coef * exponent

def predict(X, params, priors):
    preds = []
    for x in X:
        posteriors = []
        for c, stats in params.items():
            mean, var = stats["mean"], stats["var"]
            likelihood = np.prod(gaussian_prob(x, mean, var))
            posterior = np.log(priors[c]) + np.sum(np.log(likelihood + 1e-9))
            posteriors.append(posterior)
        preds.append(np.argmax(posteriors))
    return np.array(preds)

params, priors = fit(X, y)
y_pred = predict(X, params, priors)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="bwr", edgecolors="k")
plt.title("Naive Bayes Classification")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

accuracy = np.mean(y_pred == y)
print(f"Accuracy = {accuracy}")