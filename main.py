from image import generate_diagram_hot
from LogisticRegressionModel import LogisticRegressionModel
from CrossEntropyLoss import CrossEntropyLoss
from train import train, calculate_accuracy
import numpy as np

class TestModel:
    def __init__(self, num_epochs=500):
        # Model parameters
        self.num_classes = 4  # Red, Green, Blue, Yellow
        self.input_size = 20 * 20 * 4  # Size of the flattened one-hot encoded image
        self.model = LogisticRegressionModel(self.input_size, self.num_classes)
        self.num_epochs = num_epochs

    def load_data(self, train_size=1000, test_size=200):
        # Generating training data
        x_train, y_train = generate_diagram_hot(train_size, task1_or_2=True)
        
        # Generating testing data
        x_test, y_test = generate_diagram_hot(test_size, task1_or_2=True)

        # One-hot encoding the labels for training
        y_train_encoded = np.zeros((y_train.size, self.num_classes))
        y_train_encoded[np.arange(y_train.size), y_train] = 1

        # One-hot encoding the labels for testing
        y_test_encoded = np.zeros((y_test.size, self.num_classes))
        y_test_encoded[np.arange(y_test.size), y_test] = 1

        return x_train, y_train_encoded, x_test, y_test_encoded

    def run(self):
        # Load data
        x_train, y_train, x_test, y_test = self.load_data()

        # Define loss function and other parameters
        loss_fn = CrossEntropyLoss()  # Assuming you have this defined somewhere
        learning_rate = 0.001
        reg_lambda = 0.01  # Regularization parameter

        # Train the model
        train_accuracies, val_accuracies = train(
            self.model, loss_fn, x_train, y_train, x_test, y_test,
            self.num_epochs, learning_rate, reg_lambda
        )

        # Test the model
        y_test_pred = self.model.predict(x_test)
        test_accuracy = calculate_accuracy(y_test, y_test_pred)
        print(f"Test accuracy: {test_accuracy}")

        return train_accuracies, val_accuracies, test_accuracy

# Running the test
test_model = TestModel()
train_accuracies, val_accuracies, test_accuracy = test_model.run()

