# import matplotlib.pyplot as plt

import numpy as np



def calculate_accuracy(y_true, y_pred):
    # Convert predictions to label index
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    correct_predictions = np.sum(y_pred_labels == y_true_labels)
    accuracy = correct_predictions / len(y_true)
    return accuracy

def train(model, loss_fn, x_train, y_train, x_val, y_val, epochs, learning_rate, reg_lambda):
    train_accuracies = []
    val_accuracies = []
    best_loss = float('inf')
    patience = 50
    wait = 0
    index_to_color = {0: 'Red', 1: 'Green', 2: 'Blue', 3: 'Yellow'}  # Color mapping

    for epoch in range(epochs):
        # Training step
        y_pred = model.predict(x_train)
        dw, db = loss_fn.compute_gradient(x_train, y_train, y_pred)
        dw += reg_lambda * model.weights  # L2 regularization
        model.update_parameters(dw, db, learning_rate)

        # Monitoring the performance on validation set
        y_val_pred = model.predict(x_val)
        val_loss = loss_fn.compute_loss(y_val, y_val_pred)

        # Every 100 epochs, check and print a subset of predictions
        if epoch % 100 == 0:
            subset_indices = np.random.choice(len(x_train), 10, replace=False)
            x_subset = x_train[subset_indices]
            y_subset_true = y_train[subset_indices]

            # Predict on this subset
            y_subset_pred = model.predict(x_subset)
            predicted_labels = np.argmax(y_subset_pred, axis=1)

            # Print predicted and actual wire colors
            for i in range(len(subset_indices)):
                actual_color = index_to_color[np.argmax(y_subset_true[i])]
                predicted_color = index_to_color[predicted_labels[i]]
                print(f"Epoch {epoch}: Model predicted {predicted_color}, actual wire: {actual_color}")

            # Print validation loss
            print(f'Epoch {epoch}, Validation Loss: {val_loss}')

        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

        # Record accuracies
        y_train_pred = model.predict(x_train)
        train_accuracy = calculate_accuracy(y_train, y_train_pred)
        train_accuracies.append(train_accuracy)

        y_val_pred = model.predict(x_val)
        val_accuracy = calculate_accuracy(y_val, y_val_pred)
        val_accuracies.append(val_accuracy)

    return train_accuracies, val_accuracies



    # epochs_to_plot = list(range(0, epochs, 50))
    # plt.figure(figsize=(10, 6))
    # plt.plot(epochs_to_plot, train_accuracies, label='Training Accuracy')
    # plt.plot(epochs_to_plot, val_accuracies, label='Validation Accuracy')
    # plt.title('Training and Validation Accuracy over Epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()

            
