from image import generate_diagram, generate_diagram_hot
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
    
    def plot_weights(self):
        # Plot the weights as a 20x20 image
        vals = []
        w = self.w.flatten()
        w = w[1:]
        w = w.reshape(20, 20, 4)
        for v in w:
            for v2 in v:
                vals.append(np.average(v2))

        vals = np.array(vals)
        vals = vals.reshape(20, 20)
        print(vals)
        plt.imshow(vals)
        plt.show()


X, y = generate_diagram_hot(5000)

X_test, y_test = generate_diagram_hot(5000)


def train_and_evaluate(l):
    avg = 0
    for _ in range(10):
        model = LogRegModel()
        model.train(X, y, epochs=100, lr=l, lambda_reg=0)
        avg += model.evaluate(X_test, y_test)
    print("Progress:", l)
    return avg / 10

if __name__ == '__main__':
    lr = np.linspace(0.001, 0.1, 101)
    acc = []

    with Pool() as p:
        acc = p.map(train_and_evaluate, lr)

    plt.plot(lr, acc)
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.title("Learning Rate vs Accuracy")
    plt.savefig("images/lr.png")



# -----------------------------------------------------
# # accuracy vs number of training examples

# def train_and_evaluate(n):
#     model = LogRegModel()
#     model.train(X[:int(n)], y[:int(n)], epochs=100, lr=0.05, lambda_reg=0)
#     print("Progress:", n)
#     return model.evaluate(X_test, y_test)

# if __name__ == '__main__':
#     num_examples = np.linspace(1, 5000, 100)
#     acc = []

#     with Pool() as p:
#         acc = p.map(train_and_evaluate, num_examples)

#     plt.plot(num_examples, acc)
#     plt.xlabel("Number of Training Examples")
#     plt.ylabel("Accuracy")
#     plt.title("Number of Training Examples vs Accuracy")
#     plt.savefig("images/num_examples.png")

