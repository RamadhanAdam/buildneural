"""
main.py
~~~~~~~

Main script to train a neural network on the MNIST digit dataset.
"""

import mnist_loader
import network

# Load MNIST data
print("Loading MNIST data...")
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

print(f"Training data: {len(training_data)} images")
print(f"Test data: {len(test_data)} images")
print()

# Create network: 784 inputs, 30 hidden neurons, 10 outputs
print("Creating neural network [784, 30, 10]...")
net = network.Network([784, 30, 10])

# Train it
print("Starting training...")
print("(This will take a few minutes)")
print()

net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)

print()
print("Training complete!")