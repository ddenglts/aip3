import numpy as np
import matplotlib.pyplot as plt



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
    patience = 50  # Number of epochs to wait after last time validation loss improved.
    wait = 0
    for epoch in range(epochs):
        y_pred = model.predict(x_train)
        dw, db = loss_fn.compute_gradient(x_train, y_train, y_pred)
        dw += reg_lambda * model.weights  # L2 regularization
        model.update_parameters(dw, db, learning_rate)

        # Monitoring the performance on validation set
        y_val_pred = model.predict(x_val)
        val_loss = loss_fn.compute_loss(y_val, y_val_pred)
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Validation Loss: {val_loss}')
    
            y_train_pred = model.predict(x_train)
            train_accuracy = calculate_accuracy(y_train, y_train_pred)
            train_accuracies.append(train_accuracy)

            # Calculate and record validation accuracy
            y_val_pred = model.predict(x_val)
            val_accuracy = calculate_accuracy(y_val, y_val_pred)
            val_accuracies.append(val_accuracy)
        
        
        if val_loss < best_loss:
            best_loss = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
    epochs_to_plot = list(range(0, epochs, 50))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_to_plot, train_accuracies, label='Training Accuracy')
    plt.plot(epochs_to_plot, val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    return train_accuracies, val_accuracies

            
