from image import generate_diagram, generate_diagram_hot
from LogisticRegressionModel import *
from CrossEntropyLoss import *
from train import *
import numpy as np

def convert_to_int_labels(y_bool):
    return y_bool.astype(int)

def one_hot_encode(y, num_classes):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

# Data Preparation
x_train, y_train_raw = generate_diagram_hot(5000)  # Generate training data
x_val, y_val_raw = generate_diagram_hot(5000)      # Generate validation data

# Convert boolean labels to integer labels
y_train_int = convert_to_int_labels(y_train_raw)
y_val_int = convert_to_int_labels(y_val_raw)

# Convert the integer labels to one-hot encoded format
y_train = one_hot_encode(y_train_int, 4)
y_val = one_hot_encode(y_val_int, 4)

# Model Training
model = LogisticRegressionModel(1600, 4)
loss_fn = CrossEntropyLoss()
train_accuracies, val_accuracies = train(model, loss_fn, x_train, y_train, x_val, y_val, epochs=500, learning_rate=0.91, reg_lambda=0)
y_train_pred = model.predict(x_train)
train_accuracy = calculate_accuracy(y_train, y_train_pred)
print(f'Training Accuracy: {train_accuracy:.2f}')

y_val_pred = model.predict(x_val)
val_accuracy = calculate_accuracy(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy:.2f}')
