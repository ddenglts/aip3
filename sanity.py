from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from image import generate_diagram

# Load the iris dataset
data = generate_diagram(1000)
X = data[0]
y = data[1]


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)
# Compute the accuracy
num_correct = 0
for i, _ in enumerate(predictions):
    if predictions[i] == y_test[i]:
        num_correct += 1

print("Accuracy on test data:", num_correct / len(predictions))
