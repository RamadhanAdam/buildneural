"""
layers.py
~~~~~~~~~

Small NumPy layers for building a convolutional neural network from scratch.
The code favors readability over speed, so each operation is easy to inspect.
"""

import numpy as np


class Conv2D:
    """
    A simple 2D convolution layer for one grayscale image.

    Input shape:  (height, width)
    Output shape: (new_height, new_width, num_filters)

    This layer uses "valid" convolution, meaning the filter only slides where it
    fully fits inside the image. No padding is added around the image.
    """

    def __init__(self, num_filters, filter_size, learning_rate=0.01, seed=42):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.learning_rate = learning_rate

        rng = np.random.default_rng(seed)

        # He-style scaling keeps the starting values from being too tiny or too wild.
        scale = np.sqrt(2.0 / (filter_size * filter_size))
        self.filters = rng.normal(
            0.0, scale, (num_filters, filter_size, filter_size)
        )
        self.biases = np.zeros(num_filters)

    def iterate_regions(self, image):
        """
        Yield every small patch where the filter will be placed.

        For an 8x8 image and a 3x3 filter, this gives 6x6 patches.
        """
        height, width = image.shape
        for row in range(height - self.filter_size + 1):
            for col in range(width - self.filter_size + 1):
                region = image[
                    row : row + self.filter_size,
                    col : col + self.filter_size,
                ]
                yield region, row, col

    def forward(self, image):
        """Slide each filter over the image and produce feature maps."""
        self.last_input = image
        height, width = image.shape
        output = np.zeros(
            (height - self.filter_size + 1, width - self.filter_size + 1, self.num_filters)
        )

        for region, row, col in self.iterate_regions(image):
            # One patch meets every filter. Each result becomes one output pixel.
            output[row, col] = np.sum(region * self.filters, axis=(1, 2)) + self.biases

        return output

    def backward(self, d_loss_d_output):
        """
        Move the loss gradient backward through the convolution.

        d_loss_d_output tells us how much each feature-map value affected the
        final loss. We use it to update filters and to pass blame back to the
        original image pixels.
        """
        d_loss_d_filters = np.zeros(self.filters.shape)
        d_loss_d_biases = np.zeros(self.biases.shape)
        d_loss_d_input = np.zeros(self.last_input.shape)

        for region, row, col in self.iterate_regions(self.last_input):
            for filter_index in range(self.num_filters):
                gradient = d_loss_d_output[row, col, filter_index]

                # If this patch increased the loss, nudge this filter away from it.
                d_loss_d_filters[filter_index] += gradient * region
                d_loss_d_biases[filter_index] += gradient

                # The same filter also tells earlier layers which pixels mattered.
                d_loss_d_input[
                    row : row + self.filter_size,
                    col : col + self.filter_size,
                ] += gradient * self.filters[filter_index]

        # Gradient descent: step in the opposite direction of the error.
        self.filters -= self.learning_rate * d_loss_d_filters
        self.biases -= self.learning_rate * d_loss_d_biases

        return d_loss_d_input


class ReLU:
    """Keep positive signals and shut off negative signals."""

    def forward(self, input_data):
        self.last_input = input_data
        return np.maximum(0, input_data)

    def backward(self, d_loss_d_output):
        d_loss_d_input = d_loss_d_output.copy()

        # Anything that was shut off during the forward pass cannot pass gradient.
        d_loss_d_input[self.last_input <= 0] = 0
        return d_loss_d_input


class MaxPool2D:
    """
    Downsample feature maps by keeping the strongest value in each small window.

    With a 2x2 pool, a 6x6 feature map becomes 3x3.
    """

    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def iterate_regions(self, image):
        """Yield each pooling window without overlap."""
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
        """Keep the maximum value from each pooling window."""
        self.last_input = input_data
        height, width, num_filters = input_data.shape
        output = np.zeros((height // self.pool_size, width // self.pool_size, num_filters))

        for region, row, col in self.iterate_regions(input_data):
            output[row, col] = np.amax(region, axis=(0, 1))

        return output

    def backward(self, d_loss_d_output):
        """
        Send the gradient back only to the value that won each pooling window.

        Max pooling throws away the smaller values, so only the max value receives
        credit or blame during backpropagation.
        """
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
    """Turn stacked feature maps into one long vector for the dense layer."""

    def forward(self, input_data):
        self.last_shape = input_data.shape
        return input_data.flatten()

    def backward(self, d_loss_d_output):
        # Backprop needs the original image-like shape again.
        return d_loss_d_output.reshape(self.last_shape)


class Dense:
    """A fully connected layer: every input value connects to every output score."""

    def __init__(self, input_len, output_len, learning_rate=0.01, seed=42):
        self.learning_rate = learning_rate
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, np.sqrt(2.0 / input_len), (input_len, output_len))
        self.biases = np.zeros(output_len)

    def forward(self, input_data):
        """Compute class scores from the flattened features."""
        self.last_input = input_data
        return np.dot(input_data, self.weights) + self.biases

    def backward(self, d_loss_d_output):
        """Update weights and pass the gradient back to the previous layer."""
        d_loss_d_weights = np.outer(self.last_input, d_loss_d_output)
        d_loss_d_biases = d_loss_d_output
        d_loss_d_input = np.dot(self.weights, d_loss_d_output)

        self.weights -= self.learning_rate * d_loss_d_weights
        self.biases -= self.learning_rate * d_loss_d_biases

        return d_loss_d_input


class SoftmaxCrossEntropy:
    """
    Convert raw scores into probabilities and measure how wrong the answer is.

    Softmax gives probabilities. Cross-entropy punishes confident wrong answers
    more than uncertain wrong answers.
    """

    def forward(self, logits, label):
        # Subtracting max(logits) keeps exp() numerically stable.
        shifted_logits = logits - np.max(logits)
        exp_values = np.exp(shifted_logits)
        self.probabilities = exp_values / np.sum(exp_values)
        self.label = label

        return -np.log(self.probabilities[label] + 1e-12)

    def backward(self):
        # For softmax + cross-entropy, the gradient is prediction - target.
        gradient = self.probabilities.copy()
        gradient[self.label] -= 1
        return gradient
