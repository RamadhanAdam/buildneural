"""
cnn.py
~~~~~~

A tiny convolutional neural network made from our own layers in layers.py.
"""

import numpy as np

from layers import Conv2D, Dense, Flatten, MaxPool2D, ReLU, SoftmaxCrossEntropy


class TinyCNN:
    """
    The whole CNN in one place.

    Shape flow:
    8x8 image -> 6x6x4 features -> 3x3x4 pooled features -> 36 values -> 3 scores
    """

    def __init__(self, learning_rate=0.01):
        # Four filters means the network can learn four small visual detectors.
        self.conv = Conv2D(num_filters=4, filter_size=3, learning_rate=learning_rate)
        self.relu = ReLU()
        self.pool = MaxPool2D(pool_size=2)
        self.flatten = Flatten()
        self.dense = Dense(input_len=3 * 3 * 4, output_len=3, learning_rate=learning_rate)
        self.loss = SoftmaxCrossEntropy()

    def forward(self, image):
        """Run one image through every layer and return raw class scores."""
        output = self.conv.forward(image)
        output = self.relu.forward(output)
        output = self.pool.forward(output)
        output = self.flatten.forward(output)
        return self.dense.forward(output)

    def predict(self, image):
        """Return the class with the highest score."""
        logits = self.forward(image)
        return int(np.argmax(logits))

    def train_one(self, image, label):
        """Train on one image, then return its loss."""
        logits = self.forward(image)
        loss = self.loss.forward(logits, label)

        # The backward pass walks through the same layers in reverse order.
        gradient = self.loss.backward()
        gradient = self.dense.backward(gradient)
        gradient = self.flatten.backward(gradient)
        gradient = self.pool.backward(gradient)
        gradient = self.relu.backward(gradient)
        self.conv.backward(gradient)

        return loss

    def evaluate(self, images, labels):
        """Measure accuracy on a small set of images."""
        predictions = [self.predict(image) for image in images]
        return np.mean(np.array(predictions) == labels)
