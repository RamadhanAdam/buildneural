"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network. Gradients are calculated
using backpropagation.
"""

import random
import numpy as np


class Network:
    def __init__(self, sizes):
        """
        Initialize the neural network.
        
        Args:
            sizes: List of layer sizes. E.g., [784, 30, 10] creates:
                   - 784 input neurons
                   - 30 hidden neurons  
                   - 10 output neurons
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        # Random weights and biases - network knows nothing yet
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                       for x, y in zip(sizes[:-1], sizes[1:])]
    
    def feedforward(self, a):
        """
        Return the output of the network for input 'a'.
        Push input through each layer.
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        
        Args:
            training_data: List of tuples (x, y) representing inputs and desired outputs
            epochs: Number of times to go through entire training set
            mini_batch_size: Size of mini-batches (typically 10)
            eta: Learning rate (typically 3.0)
            test_data: Optional test data to evaluate progress
        """
        n = len(training_data)
        
        for j in range(epochs):
            # Shuffle data each epoch
            random.shuffle(training_data)
            
            # Split into mini-batches
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            
            # Learn from each mini-batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            # Check progress on test data
            if test_data:
                correct = self.evaluate(test_data)
                print(f"Epoch {j}: {correct} / {len(test_data)}")
            else:
                print(f"Epoch {j} complete")
    
    def update_mini_batch(self, mini_batch, eta):
        """
        Update network weights and biases by applying gradient descent
        using backpropagation to a single mini batch.
        """
        # Start with zero gradients
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # For each image in this batch
        for x, y in mini_batch:
            # Compute how to adjust weights for THIS image
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            
            # Accumulate adjustments
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # Apply average adjustment to all weights/biases
        self.weights = [w - (eta/len(mini_batch)) * nw 
                       for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch)) * nb 
                      for b, nb in zip(self.biases, nabla_b)]
    
    def backprop(self, x, y):
        """
        Return a tuple (nabla_b, nabla_w) representing the gradient
        for the cost function. Error flows backward through the network.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Forward pass - store activations
        activation = x
        activations = [x]  # List to store all activations, layer by layer
        zs = []  # List to store all z vectors, layer by layer
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # Backward pass - compute gradients
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        # Propagate error backward through layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
        return (nabla_b, nabla_w)
    
    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural network 
        outputs the correct result.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives for the output activations.
        How wrong was the final answer?
        """
        return (output_activations - y)


# Helper functions
def sigmoid(z):
    """The sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))