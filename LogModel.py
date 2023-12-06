from image import generate_diagram
import numpy as np

#USING SGD:

class BinaryPerceptronSGD:
    def __init__(self, n):
        # Initialize the weights, including the bias as the first weight w_0
        self.weights = np.random.randn(n + 1)
    
    def predict(self, x):
        # Compute the weighted sum of inputs, including the bias w_0*x_0 where x_0 is always 1
        weighted_sum = np.dot(self.weights, x)
        # Apply the step function
        print("Weighted sum:", weighted_sum)
        return 1 if weighted_sum >= 0 else 0
    
    def train(self, X, y, epochs=1, learning_rate=1.0):
        # Add a column of ones to X to account for the bias weight w_0
        X = np.insert(X, 0, 1, axis=1)
        # X is now a matrix of input vectors with a leading 1 for the bias, y is a vector of true labels
        for epoch in range(epochs):
            for i in range(len(X)):
                # Make a prediction for the current input vector
                prediction = self.predict(X[i])
                # Update the weights based on the prediction error
                # This is the perceptron learning rule, which is a form of SGD
                error = prediction - y[i]
                self.weights -= learning_rate * error * X[i]


NUM_EPOCHS = 100
LEARNING_RATE = 0.1
NUM_TRAINING_IMAGES = 500
NUM_TEST_IMAGES = 100


images_training_data = generate_diagram(NUM_TRAINING_IMAGES)
training_images = images_training_data[0]
training_labels = images_training_data[1]

model = BinaryPerceptronSGD(training_images.shape[1])
model.train(training_images, training_labels, epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)

images_test_data = generate_diagram(NUM_TEST_IMAGES)
test_images = images_test_data[0]
test_images = np.insert(test_images, 0, 1, axis=1)
test_labels = images_test_data[1]

num_correct = 0
for i in range(len(test_images)):
    prediction = model.predict(test_images[i])
    if prediction == test_labels[i]:
        num_correct += 1

print("Accuracy on test data:", num_correct / len(test_images))

