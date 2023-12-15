import numpy as np
from MLRMimage import *

class MultiClassLogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        m = y_true.shape[1]
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss

    def fit(self, X, y):
        num_features, num_samples = X.shape
        num_classes = y.shape[0]

        self.weights = np.random.rand(num_classes, num_features)
        self.bias = np.zeros((num_classes, 1))

        for i in range(self.epochs):
            Z = np.dot(self.weights, X) + self.bias
            A = self.softmax(Z)

            cost = self.cross_entropy_loss(y, A)

            dW = np.dot(A - y, X.T) / num_samples
            db = np.sum(A - y, axis=1, keepdims=True) / num_samples

            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db

    def predict(self, X):
        Z = np.dot(self.weights, X) + self.bias
        A = self.softmax(Z)
        return np.argmax(A, axis=0)

    def accuracy(self, y_true, y_pred):
        # Convert one-hot encoded y_true to class labels
        y_true_labels = np.argmax(y_true, axis=1)
        return np.mean(y_true_labels == y_pred)




if __name__ == "__main__":
    # Load the data
    X_train, y_train = generate_diagram_nonlinear3(5000)
    X_test, y_test = generate_diagram_nonlinear3(5000)

    # Initialize the model
    model = MultiClassLogisticRegression(learning_rate=0.34, epochs=500)

    # Train the model
    model.fit(X_train.T, y_train.T)

    # Evaluate the model

    y_pred = model.predict(X_train.T)
    print("Training Accuracy:", model.accuracy(y_train, y_pred))
    
    y_pred = model.predict(X_test.T)
    print("Testing Accuracy:", model.accuracy(y_test, y_pred))