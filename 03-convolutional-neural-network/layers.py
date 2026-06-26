"""
layers.py
~~~~~~~~~

Small NumPy layers for building a convolutional neural network from scratch.
The code favors readability over speed so each operation is easy to inspect.
"""

import numpy as np


class Conv2D:
    def __init__(self, num_filters, filter_size, learning_rate=0.01, seed=42):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.learning_rate = learning_rate

        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (filter_size * filter_size))
        self.filters = rng.normal(
            0.0, scale, (num_filters, filter_size, filter_size)
        )
        self.biases = np.zeros(num_filters)

    def iterate_regions(self, image):
        height, width = image.shape
        for row in range(height - self.filter_size + 1):
            for col in range(width - self.filter_size + 1):
                region = image[
                    row : row + self.filter_size,
                    col : col + self.filter_size,
                ]
                yield region, row, col

    def forward(self, image):
        self.last_input = image
        height, width = image.shape
        output = np.zeros(
            (height - self.filter_size + 1, width - self.filter_size + 1, self.num_filters)
        )

        for region, row, col in self.iterate_regions(image):
            output[row, col] = np.sum(region * self.filters, axis=(1, 2)) + self.biases

        return output

    def backward(self, d_loss_d_output):
        d_loss_d_filters = np.zeros(self.filters.shape)
        d_loss_d_biases = np.zeros(self.biases.shape)
        d_loss_d_input = np.zeros(self.last_input.shape)

        for region, row, col in self.iterate_regions(self.last_input):
            for filter_index in range(self.num_filters):
                gradient = d_loss_d_output[row, col, filter_index]
                d_loss_d_filters[filter_index] += gradient * region
                d_loss_d_biases[filter_index] += gradient
                d_loss_d_input[
                    row : row + self.filter_size,
                    col : col + self.filter_size,
                ] += gradient * self.filters[filter_index]

        self.filters -= self.learning_rate * d_loss_d_filters
        self.biases -= self.learning_rate * d_loss_d_biases

        return d_loss_d_input


class ReLU:
    def forward(self, input_data):
        self.last_input = input_data
        return np.maximum(0, input_data)

    def backward(self, d_loss_d_output):
        d_loss_d_input = d_loss_d_output.copy()
        d_loss_d_input[self.last_input <= 0] = 0
        return d_loss_d_input


class MaxPool2D:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def iterate_regions(self, image):
        height, width, num_filters = image.shape
        new_height = height // self.pool_size
        new_width = width // self.pool_size

        for row in range(new_height):
            for col in range(new_width):
                region = image[
                    row * self.pool_size : (row + 1) * self.pool_size,
                    col * self.pool_size : (col + 1) * self.pool_size,
                ]
                yield region, row, col

    def forward(self, input_data):
        self.last_input = input_data
        height, width, num_filters = input_data.shape
        output = np.zeros((height // self.pool_size, width // self.pool_size, num_filters))

        for region, row, col in self.iterate_regions(input_data):
            output[row, col] = np.amax(region, axis=(0, 1))

        return output

    def backward(self, d_loss_d_output):
        d_loss_d_input = np.zeros(self.last_input.shape)

        for region, row, col in self.iterate_regions(self.last_input):
            height, width, num_filters = region.shape
            max_values = np.amax(region, axis=(0, 1))

            for i in range(height):
                for j in range(width):
                    for filter_index in range(num_filters):
                        if region[i, j, filter_index] == max_values[filter_index]:
                            d_loss_d_input[
                                row * self.pool_size + i,
                                col * self.pool_size + j,
                                filter_index,
                            ] = d_loss_d_output[row, col, filter_index]

        return d_loss_d_input


class Flatten:
    def forward(self, input_data):
        self.last_shape = input_data.shape
        return input_data.flatten()

    def backward(self, d_loss_d_output):
        return d_loss_d_output.reshape(self.last_shape)


class Dense:
    def __init__(self, input_len, output_len, learning_rate=0.01, seed=42):
        self.learning_rate = learning_rate
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, np.sqrt(2.0 / input_len), (input_len, output_len))
        self.biases = np.zeros(output_len)

    def forward(self, input_data):
        self.last_input = input_data
        return np.dot(input_data, self.weights) + self.biases

    def backward(self, d_loss_d_output):
        d_loss_d_weights = np.outer(self.last_input, d_loss_d_output)
        d_loss_d_biases = d_loss_d_output
        d_loss_d_input = np.dot(self.weights, d_loss_d_output)

        self.weights -= self.learning_rate * d_loss_d_weights
        self.biases -= self.learning_rate * d_loss_d_biases

        return d_loss_d_input


class SoftmaxCrossEntropy:
    def forward(self, logits, label):
        shifted_logits = logits - np.max(logits)
        exp_values = np.exp(shifted_logits)
        self.probabilities = exp_values / np.sum(exp_values)
        self.label = label

        return -np.log(self.probabilities[label] + 1e-12)

    def backward(self):
        gradient = self.probabilities.copy()
        gradient[self.label] -= 1
        return gradient
