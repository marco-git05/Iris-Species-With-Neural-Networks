from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from neural_network import NeuralNetwork

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target.reshape(-1, 1) # Ensure y is column vector for encoder

# Onehot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Splits 
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create a neural network object
nn = NeuralNetwork(input=X_train, target=y_train, hidden=5, learning_rate=0.1, iterations=10000)

# Train the neural network
nn.train()

predictions = nn.predict(X_test)

# Computes the accuracy using probabilities
def compute_accuracy(predictions, targets):
    pred_labels = predictions.argmax(axis=1)
    true_labels = targets.argmax(axis=1)
    accuracy = (pred_labels == true_labels).mean()
    return accuracy

# Prints the accuracy of the neural network to the terminal
print("Accuracy on test set:", compute_accuracy(predictions, y_test))