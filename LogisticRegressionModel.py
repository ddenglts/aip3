import numpy as np

class LogisticRegressionModel:
    def __init__(self, input_size, num_classes):
        self.weights = np.zeros((input_size, num_classes))
        self.bias = np.zeros(num_classes)

    def predict(self, x):
        logits = np.dot(x, self.weights) + self.bias
        return self.softmax(logits)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def update_parameters(self, dw, db, learning_rate):
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db