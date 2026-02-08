"""
mnist_loader.py
~~~~~~~~~~~~~~~

A library to load the MNIST image data. Returns the data in a format
suited for use in the neural network.
"""

import pickle
import gzip
import numpy as np


def load_data():
    """
    Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    """
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    """
    Return a tuple containing (training_data, validation_data, test_data).
    
    training_data is a list of tuples (x, y):
    - x is a 784-dimensional numpy array (28x28 pixels flattened)
    - y is a 10-dimensional numpy array representing the digit (0-9)
    
    validation_data and test_data are lists of tuples (x, y):
    - x is a 784-dimensional numpy array
    - y is the actual digit (not vectorized)
    """
    tr_d, va_d, te_d = load_data()
    
    # Training data: vectorize both inputs and outputs
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    
    # Validation data: vectorize inputs only
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    
    # Test data: vectorize inputs only
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    
    return (training_data, validation_data, test_data)


def vectorized_result(j):
    """
    Return a 10-dimensional unit vector with a 1.0 in the jth position
    and zeroes elsewhere.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e