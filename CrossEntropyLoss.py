import numpy as np

class CrossEntropyLoss:
    @staticmethod
    def compute_loss(y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-15)) / m

    @staticmethod
    def compute_gradient(x, y_true, y_pred):
        m = x.shape[0]
        dw = np.dot(x.T, (y_pred - y_true)) / m
        db = np.sum(y_pred - y_true, axis=0) / m
        return dw, db