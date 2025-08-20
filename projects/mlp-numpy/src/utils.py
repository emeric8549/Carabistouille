import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def get_moons(n_samples=1000, noise=0.2, seed=None):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    return X, y.reshape(-1, 1)


def get_circles(n_samples=1000, noise=0.1, seed=None):
    X, y = make_circles(n_samples=n_samples, noise=noise, random_state=seed)
    return X, y.reshape(-1, 1)


def get_data(n_samples=1000, n_features=5, n_redundant=0, n_informative=5, n_clusters_per_class=1, n_classes=3, seed=None):
    X, y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_redundant=n_redundant,
                               n_informative=n_informative,
                               n_clusters_per_class=n_clusters_per_class,
                               n_classes=n_classes,
                               random_state=seed)


    return X, np.eye(n_classes)[y]


def split_data(X, y, test_size):
    indices = np.arange(len(X))
    test_limit = int(len(X) * test_size)
    train_indices = indices[test_limit:]
    test_indices = indices[:test_limit]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, y_train, X_test, y_test


def draw_boundary(X, y, model, filename):
    filepath = "plots/" + filename
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 200),
                         np.linspace(x2_min, x2_max, 200))

    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = model.forward(grid).reshape(xx.shape)
    pred = model.predict(grid).reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, pred, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.contour(xx, yy, Z, levels=[0.5], colors="black", linewidths=2)

    sc0 = plt.scatter(X[y[:, 0]==0, 0], X[y[:, 0]==0, 1], color="blue", marker="+", label="Class 0")
    sc1 = plt.scatter(X[y[:, 0]==1, 0], X[y[:, 0]==1, 1], color="red", marker="+", label="Class 1")

    decision_boundary = mlines.Line2D([], [], color="black", linewidth=2, label="Decision boundary") # trick to add decision boundary to legend

    plt.legend(handles=[sc0, sc1, decision_boundary])
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Decision Boundary MLP")
    plt.savefig(filepath)
    plt.close()