from image import generate_diagram_hot
from LogisticRegressionModel import LogisticRegressionModel
from CrossEntropyLoss import CrossEntropyLoss
from train import train, calculate_accuracy
import numpy as np

def one_hot_encode(y, num_classes):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

def prepare_data(samples, num_classes):
    x_data, y_data = [], []
    color_to_index = {'Red': 0, 'Green': 1, 'Blue': 2, 'Yellow': 3}  # Map color names to indices

    for _ in range(samples):
        diagram, wire_to_cut_array = generate_diagram_hot(1, True)
        x_data.append(diagram.flatten())  # Flatten the diagram

        if wire_to_cut_array.size == 1:
            wire_to_cut_color = wire_to_cut_array.item()  # Convert array to string
            wire_to_cut_index = color_to_index[wire_to_cut_color]  # Convert color name to index
            y_data.append(wire_to_cut_index)
        else:
            # Handle unexpected format
            raise ValueError("Unexpected format for wire color")

    return np.array(x_data), one_hot_encode(np.array(y_data), num_classes)


# Data Preparation
x_train, y_train = prepare_data(5000, 4)  # Generate training data
x_val, y_val = prepare_data(5000, 4)      # Generate validation data

# Model Training
model = LogisticRegressionModel(1600, 4)
loss_fn = CrossEntropyLoss()
train_accuracies, val_accuracies = train(model, loss_fn, x_train, y_train, x_val, y_val, epochs=500, learning_rate=0.05, reg_lambda=0)

# Calculate and Print Accuracies
y_train_pred = model.predict(x_train)
train_accuracy = calculate_accuracy(y_train, y_train_pred)
print(f'Training Accuracy: {train_accuracy:.2f}')

y_val_pred = model.predict(x_val)
val_accuracy = calculate_accuracy(y_val, y_val_pred)
print(f'Validation Accuracy: {val_accuracy:.2f}')
