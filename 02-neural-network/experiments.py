"""
experiments.py
~~~~~~~~~~~~~~

Try different network configurations and hyperparameters.
Uncomment the experiment you want to run.
"""

import mnist_loader
import network

# Load data once
print("Loading MNIST data...")
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print(f"Loaded {len(training_data)} training images\n")


# EXPERIMENT 1: Baseline
print("EXPERIMENT 1: Baseline [784, 30, 10], eta=3.0")
net = network.Network([784, 30, 10])
net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)


# EXPERIMENT 2: Learning rate too slow
# print("\nEXPERIMENT 2: eta=0.001 (too slow)")
# net = network.Network([784, 30, 10])
# net.SGD(training_data, epochs=30, mini_batch_size=10, eta=0.001, test_data=test_data)


# EXPERIMENT 3: Learning rate too fast
# print("\nEXPERIMENT 3: eta=100 (too fast)")
# net = network.Network([784, 30, 10])
# net.SGD(training_data, epochs=30, mini_batch_size=10, eta=100, test_data=test_data)


# EXPERIMENT 4: More neurons
# print("\nEXPERIMENT 4: [784, 100, 10]")
# net = network.Network([784, 100, 10])
# net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)


# EXPERIMENT 5: Deeper network
# print("\nEXPERIMENT 5: [784, 30, 30, 10]")
# net = network.Network([784, 30, 30, 10])
# net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)