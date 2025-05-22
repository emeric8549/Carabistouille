import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification

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
