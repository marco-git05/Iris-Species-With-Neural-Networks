import numpy as np

class NeuralNetwork:

    def __init__(self, input, target, hidden, learning_rate, iterations):
        self.X = input
        self.Y = target
        self.learning_rate = learning_rate
        self.iterations = iterations

        input_size = self.X.shape[1]
        output_size = self.Y.shape[1]

        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden) * 0.01
        self.b1 = np.zeros((1, hidden))
        self.W2 = np.random.randn(hidden, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid_function(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        s = self.sigmoid_function(z)
        return s * (1 - s)

    def softmax(z):
        return np.exp(z) / sum(np.exp(z))

    def feed_forward(self):
        self.Z1 = np.dot(self.W1, self.X) + self.b1
        self.A1 = self.sigmoid_function(self.Z1)
        self.Z2 = np.dot(self.W2, self.X) + self.b2
        self.fA2 = self.softmax(self.Z2)

    def backpropagation(self):
        m = self.Y.shape[0]

        self.dW2 = -2 * (self.target - self.approx - self.b1) * np.transpose(self.X)
        self.db2 = -2 * (self.target - self.approx - self.b1)
        self.dW1 = -2 * (self.target - self.approx - self.b2) * np.transpose(self.X)
        self.db1 =  -2 * (self.target - self.approx - self.b2)

        self.W1 = self.W1 - self.learning_rate * self.dW1
        self.b1 = self.b1 - self.learning_rate * self.db1 
        self.W2 = self.W2 - self.learning_rate * self.dW2
        self.b2 = self.b2 - self.learning_rate * self.db2 

    def train(self):
        for i in range(self.iterations):
            self.feed_forward()
            self.backpropigation()

    def predict(self, X):
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.softmax(Z2)
        return A2