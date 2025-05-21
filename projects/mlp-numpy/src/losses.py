import numpy as np

def BCE_loss(y, y_pred):
    loss = - (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return np.mean(loss)


def cross_entropy_loss(y, y_pred):
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    loss = -np.sum(y * np.log(y_pred), axis=1)
    return np.mean(loss)
