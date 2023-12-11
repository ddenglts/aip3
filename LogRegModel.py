from image import generate_diagram, generate_diagram_hot
import numpy as np
import matplotlib.pyplot as plt

#USING SGD:

class LogRegModel:
    def __init__(self, n):
        # Initialize the weights, including the bias as the first weight w_0
        self.weights = np.random.randn(n + 1)
        self.losses: list[float] = []
    
    def predict(self, x):
        '''
        x is a vector of input values, including the bias x_0
        '''
        return 1 if self.sigmoid(self.weights, x) >= 0.5 else 0
    
    def train(self, X, y, epochs=1, learning_rate=1.0):
        '''
        X is a matrix of input vectors, including the bias x_0
        y is a vector of labels
        '''
        # Add a column of ones to X to account for the bias weight w_0
        X = np.insert(X, 0, 1, axis=1)
        # X is now a matrix of input vectors with a leading 1 for the bias, y is a vector of true labels
        for _ in range(epochs):
            epoch_loss = 0
            for j, _ in enumerate(X):
                loss = self.sigmoid(self.weights, X[j]) - y[j]
                self.weights -= learning_rate * X[j] * loss
                epoch_loss += loss
            self.losses.append(epoch_loss/len(X))

    def sigmoid(self, w, x):
        return 1 / (1 + np.exp(-np.dot(w, x)))


if __name__ == '__main__':

    NUM_EPOCHS = 10
    LEARNING_RATE = 0.01
    NUM_TRAINING_IMAGES = 1600
    NUM_TEST_IMAGES = 200

    images_training_data = generate_diagram_hot(NUM_TRAINING_IMAGES)
    training_images = images_training_data[0]
    training_labels = images_training_data[1]

    model = LogRegModel(training_images.shape[1])
    model.train(training_images, training_labels, epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)

    images_test_data = generate_diagram_hot(NUM_TEST_IMAGES)
    test_images = images_test_data[0]
    test_images = np.insert(test_images, 0, 1, axis=1)
    test_labels = images_test_data[1]

    num_correct = 0
    for i, _ in enumerate(test_images):
        prediction = model.predict(test_images[i])
        if prediction == test_labels[i]:
            num_correct += 1

    print("weights after training: ", model.weights[:10])
    print("Accuracy on test data:", num_correct / len(test_images))


    training_images = np.insert(training_images, 0, 1, axis=1)
    # overfitting? testing with trained data
    num_correct = 0
    for i, _ in enumerate(training_images):
        prediction = model.predict(training_images[i])
        if prediction == training_labels[i]:
            num_correct += 1
    
    print("Accuracy on training data:", num_correct / len(training_images))

    plt.plot(model.losses)
    plt.show()