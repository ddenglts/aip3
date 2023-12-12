# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
from image import generate_diagram

# # Load the iris dataset
# data = generate_diagram(1000)
# X = data[0]
# y = data[1]


# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train the model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Make predictions on the test set
# predictions = model.predict(X_test)
# # Compute the accuracy
# num_correct = 0
# for i, _ in enumerate(predictions):
#     if predictions[i] == y_test[i]:
#         num_correct += 1

# print("Accuracy on test data:", num_correct / len(predictions))

import numpy as np

# def generate_data(num_samples):
#     # Implement your data generation logic here
#     X = np.random.rand(num_samples, 20, 20, 4)  # Example random data
#     Y = np.random.randint(0, 2, num_samples)   # Binary labels: 0 for 'Safe', 1 for 'Dangerous'
#     return X.reshape(num_samples, -1), Y       # Reshape X to be 2D

X_train, Y_train = generate_diagram(400)
X_test, Y_test = generate_diagram(200)


input_size = 20 * 20  # Flatten the 20x20x4 input [0,0,0,0], blue: [1,0,0,0], green: [0,1,0,0]
hidden_size = 128         # Size of the hidden layer
output_size = 1           # One neuron for binary output

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))



def forward_prop(X):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return A1, A2


def compute_loss(Y, A2):
    m = Y.shape[0]
    log_loss = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    return log_loss / m



def back_prop(X, Y, A1, A2):
    m = X.shape[0]
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dZ1 = np.dot(dZ2, W2.T) * (A1 * (1 - A1))
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2



def update_params(dW1, db1, dW2, db2, learning_rate):
    global W1, b1, W2, b2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2


num_epochs = 1000
learning_rate = 0.01

for epoch in range(num_epochs):
    A1, A2 = forward_prop(X_train)
    cost = compute_loss(Y_train, A2)
    dW1, db1, dW2, db2 = back_prop(X_train, Y_train, A1, A2)
    update_params(dW1, db1, dW2, db2, learning_rate)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, loss: {cost}")


#Make predictions

def predict(X):
    _, A2 = forward_prop(X)
    predictions = A2 > 0.5
    return predictions

# Predict on test data
test_predictions = predict(X_test)
test_accuracy = np.mean(test_predictions == Y_test)
print(f"Test accuracy: {test_accuracy}")


