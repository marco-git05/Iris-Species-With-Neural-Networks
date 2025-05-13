from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

# Load Iris dataset
data = load_iris()
X = data.data
y = data.target.reshape(-1, 1) # Ensure y is column vector for encoder

# Onehot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

# Output of y_encoded
print(y_encoded[:5]) # Show first 5 encoded labels