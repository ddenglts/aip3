from image import *
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool

class LogRegModel:
    def __init__(self):
        self.w = None
        self.losses = None

    def train(self, X, y, epochs, lr, lambda_reg):
        # X is a 2d numpy array of shape (n, 1600)
        # y is a 1d numpy array of shape (n,)
        # epochs is an integer
        # lr is a float
        # lambda_reg is the regularization parameter

        # Initialize weights, w_0 = bias
        self.w = np.random.randn(X.shape[1] + 1)

        # Initialize loss array
        self.losses = []

        X = np.insert(X, 0, 1, axis=1)
        
        for _ in range(epochs):
            # Shuffle the data
            # indices = np.arange(X.shape[0])
            # np.random.shuffle(indices)
            # X = X[indices]
            # y = y[indices]

            # Iterate through each image
            for i in range(X.shape[0]):
                # Get the image and label
                x = X[i]
                label = y[i]

                # Compute the gradient
                grad = self._gradient(x, label)


                # Update the weights with L2 regularization
                self.w -= lr * (grad + lambda_reg * self.w)
            #loss = self._compute_loss(X, y)
            #self.losses.append(loss)

                

    def _gradient(self, x, label):
        return (self._sigmoid(self.w @ x) - label) * x
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    


    
    def predict(self, X, cutoff = 0.5):
        # X is a 2d numpy array of shape (n, 1600)
        X = np.insert(X, 0, 1, axis=1)
        return self._sigmoid(X @ self.w) > cutoff

    def evaluate(self, X, y, cutoff = 0.5):
        # X is a 2d numpy array of shape (n, 1600)
        # y is a 1d numpy array of shape (n,)
        return np.mean(self.predict(X, cutoff=cutoff) == y)
    
    def _compute_loss(self, X, y):
        """
        Compute the log loss.
        """
        # Convert y to int
        y = y.astype(int)

        # Compute the predictions
        z = np.dot(X, self.w)
        predictions = self._sigmoid(z)

        # Compute the log loss
        epsilon = 1e-15  # to prevent division by zero
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions)

        return loss.mean()
    
    def plot_weights(self):
        # Plot the weights as a 20x20 image
        vals = []
        w = self.w.flatten()
        w = w[1:]
        w = w.reshape(20, 20, 4)
        for v in w:
            for v2 in v:
                vals.append(np.max(v2))

        vals = np.array(vals)
        vals = vals.reshape(20, 20)
        print(vals)
        plt.imshow(vals)
        plt.show()

'''
Optimal parameters:
Learning rate: 0.05
Regularization parameter: 0
Number of epochs: 150

nl2: 2500 examples training

'''





X, y = generate_diagram_nonlinear3(5000)

X_test, y_test = generate_diagram_nonlinear3(5000)

# model = LogRegModel()
# model.train(X, y, epochs=150, lr=0.05, lambda_reg=0)
# print(model.evaluate(X, y))
# print(model.evaluate(X_test, y_test))
# plt.plot(model.losses)
# plt.xlabel("Iteration")
# plt.ylabel("Log Loss")
# plt.show()



# # -----------------------------------------------------
# accuracy vs regularization parameter
# def train_and_evaluate(l):
#     model = LogRegModel()
#     model.train(X, y, epochs=150, lr=0.05, lambda_reg=l)
#     print("Progress:", l)
#     train_acc = model.evaluate(X, y)
#     test_acc = model.evaluate(X_test, y_test)
#     return train_acc, test_acc

# if __name__ == '__main__':
#     lambda_reg = np.linspace(0, 0.1, num=100)
#     train_acc = []
#     test_acc = []

#     with Pool() as p:
#         results = p.map(train_and_evaluate, lambda_reg)
#         train_acc, test_acc = zip(*results)

#     plt.plot(lambda_reg, train_acc, label='Training Accuracy')
#     plt.plot(lambda_reg, test_acc, label='Test Accuracy')
#     plt.xlabel("Regularization Parameter")
#     plt.ylabel("Accuracy")
#     plt.title("Regularization Parameter vs Accuracy")
#     plt.legend()
#     plt.savefig("images/lambda.png")



# ----------------------------------
# def train_and_evaluate(l):
#     avg = 0
#     for _ in range(10):
#         model = LogRegModel()
#         model.train(X, y, epochs=100, lr=l, lambda_reg=0)
#         avg += model.evaluate(X_test, y_test)
#     print("Progress:", l)
#     return avg / 10

# if __name__ == '__main__':
#     lr = np.linspace(0.001, 0.1, 101)
#     acc = []

#     with Pool() as p:
#         acc = p.map(train_and_evaluate, lr)

#     plt.plot(lr, acc)
#     plt.xlabel("Learning Rate")
#     plt.ylabel("Accuracy")
#     plt.title("Learning Rate vs Accuracy")
#     plt.savefig("images/lr.png")



# -----------------------------------------------------
# accuracy vs number of training examples

def train_and_evaluate(n):
    model = LogRegModel()
    model.train(X[:int(n)], y[:int(n)], epochs=150, lr=0.02, lambda_reg=0)
    print("Progress:", n)
    train_acc = model.evaluate(X[:int(n)], y[:int(n)])
    test_acc = model.evaluate(X_test, y_test)
    return train_acc, test_acc

if __name__ == '__main__':
    num_examples = np.linspace(1, X.shape[0], 100)
    train_acc = []
    test_acc = []

    with Pool() as p:
        results = p.map(train_and_evaluate, num_examples)
        train_acc, test_acc = zip(*results)

    plt.plot(num_examples, train_acc, label='Training Accuracy')
    plt.plot(num_examples, test_acc, label='Test Accuracy')
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Accuracy")
    plt.title("Number of Training Examples vs Accuracy")
    plt.legend()
    plt.savefig("images/num_examples.png")

# -----------------------------------------------------
# accuracy vs number of TESTING examples

# def train_and_evaluate(n):
#     model = LogRegModel()
#     model.train(X, y, epochs=150, lr=0.02, lambda_reg=0)
#     print("Progress:", n)
#     train_acc = model.evaluate(X, y)
#     test_acc = model.evaluate(X_test[:int(n)], y_test[:int(n)])
#     return train_acc, test_acc

# if __name__ == '__main__':
#     num_examples = np.linspace(1, X_test.shape[0], 100)
#     train_acc = []
#     test_acc = []

#     with Pool() as p:
#         results = p.map(train_and_evaluate, num_examples)
#         train_acc, test_acc = zip(*results)

#     plt.plot(num_examples, train_acc, label='Training Accuracy')
#     plt.plot(num_examples, test_acc, label='Test Accuracy')
#     plt.xlabel("Number of Testing Examples")
#     plt.ylabel("Accuracy")
#     plt.title("Number of Testing Examples vs Accuracy")
#     plt.legend()
#     plt.savefig("images/num_examples.png")

# -----------------------------------------------------
# accuracy vs number of epochs

# def train_and_evaluate(n):
#     model = LogRegModel()
#     model.train(X, y, epochs=int(n), lr=0.05, lambda_reg=0)
#     print("Progress:", n)
#     train_acc = model.evaluate(X, y)
#     test_acc = model.evaluate(X_test, y_test)
#     return train_acc, test_acc

# if __name__ == '__main__':
#     num_epochs = np.linspace(1, 1000, 100)
#     train_acc = []
#     test_acc = []

#     with Pool() as p:
#         results = p.map(train_and_evaluate, num_epochs)
#         train_acc, test_acc = zip(*results)

#     plt.plot(num_epochs, train_acc, label='Training Accuracy')
#     plt.plot(num_epochs, test_acc, label='Test Accuracy')
#     plt.xlabel("Number of Epochs")
#     plt.ylabel("Accuracy")
#     plt.title("Number of Epochs vs Accuracy")
#     plt.legend()
#     plt.savefig("images/num_epochs.png")

#---------------------------------
# accuracy vs cutoff
# def train_and_evaluate(n):
#     model = LogRegModel()
#     model.train(X, y, epochs=150, lr=0.05, lambda_reg=0)
#     print("Progress:", n)
#     train_acc = model.evaluate(X, y, cutoff=n)
#     test_acc = model.evaluate(X_test, y_test, cutoff=n)
#     return train_acc, test_acc

# if __name__ == '__main__':
#     num_epochs = np.linspace(0, 1, 100)
#     train_acc = []
#     test_acc = []

#     with Pool() as p:
#         results = p.map(train_and_evaluate, num_epochs)
#         train_acc, test_acc = zip(*results)

#     plt.plot(num_epochs, train_acc, label='Training Accuracy')
#     plt.plot(num_epochs, test_acc, label='Test Accuracy')
#     plt.xlabel("Cutoff")
#     plt.ylabel("Accuracy")
#     plt.title("Cutoff vs Accuracy")
#     plt.legend()
#     plt.savefig("images/cutoff.png")