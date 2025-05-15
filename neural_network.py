import numpy as np

class NeuralNetwork:

    def __init__(self, input, target, hidden, learning_rate, iterations):
        self.X = input
        self.Y = target
        self.learning_rate = learning_rate
        self.iterations = iterations

        input_size = self.X.shape[1]
        output_size = self.Y.shape[1]

        # Initialize weights and biases for input -> hidden
        self.W1 = np.random.randn(input_size, hidden) * 0.01
        self.b1 = np.zeros((1, hidden))

        # Initialize weights and biases for hidden -> output
        self.W2 = np.random.randn(hidden, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid_function(self, z):
        return 1 / (1 + np.exp(-z))

    # Function that returns the derivative of the sigmoid function
    def sigmoid_derivative(self, z):
        s = self.sigmoid_function(z)
        return s * (1 - s)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # Forwrd propigation
    def feed_forward(self):
        self.Z1 = np.dot(self.X, self.W1) + self.b1
        self.A1 = self.sigmoid_function(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)

    def backpropagation(self):
        m = self.Y.shape[0]

        dZ2 = self.A2 - self.Y 
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(self.X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W1 = self.W1 - self.learning_rate * dW1
        self.b1 = self.b1 - self.learning_rate * db1 
        self.W2 = self.W2 - self.learning_rate * dW2
        self.b2 = self.b2 - self.learning_rate * db2 

    def train(self):
        for i in range(self.iterations):
            self.feed_forward()
            self.backpropagation()

    def predict(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.sigmoid_function(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.softmax(Z2)
        return A2