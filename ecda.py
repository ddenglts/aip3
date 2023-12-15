import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from MLRMimage import generate_diagram_nonlinear3

# Load a fixed test set
X_test, y_test = generate_diagram_nonlinear3(5000)
y_test_labels = np.argmax(y_test, axis=1)

# Range of training sizes
training_sizes = np.arange(500, 5501, 500)
testing_accuracies = []

for size in training_sizes:
    # Load training data with the specified size
    X_train, y_train = generate_diagram_nonlinear3(size)
    y_train_labels = np.argmax(y_train, axis=1)

    # Create a pipeline with feature selection and SGDClassifier
    selector = SelectKBest(f_classif, k=25)  # Adjust 'k' based on your dataset
    model = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.01, max_iter=500, penalty='l2')
    pipeline = make_pipeline(StandardScaler(), selector, model)

    # Train the model
    pipeline.fit(X_train, y_train_labels)

    # Evaluate the model
    accuracy = pipeline.score(X_test, y_test_labels)
    testing_accuracies.append(accuracy)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(training_sizes, testing_accuracies, marker='o')
plt.title('Testing Accuracy vs Number of Training Samples')
plt.xlabel('Number of Training Samples')
plt.ylabel('Testing Accuracy')
plt.grid(True)
plt.savefig('images/ec_smallest_model.png')  # Save the plot
plt.show()
