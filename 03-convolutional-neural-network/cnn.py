"""
cnn.py
~~~~~~

A tiny convolutional neural network made from the layers in layers.py.
"""

import numpy as np

from layers import Conv2D, Dense, Flatten, MaxPool2D, ReLU, SoftmaxCrossEntropy


class TinyCNN:
    def __init__(self, learning_rate=0.01):
        self.conv = Conv2D(num_filters=4, filter_size=3, learning_rate=learning_rate)
        self.relu = ReLU()
        self.pool = MaxPool2D(pool_size=2)
        self.flatten = Flatten()
        self.dense = Dense(input_len=3 * 3 * 4, output_len=3, learning_rate=learning_rate)
        self.loss = SoftmaxCrossEntropy()

    def forward(self, image):
        output = self.conv.forward(image)
        output = self.relu.forward(output)
        output = self.pool.forward(output)
        output = self.flatten.forward(output)
        return self.dense.forward(output)

    def predict(self, image):
        logits = self.forward(image)
        return int(np.argmax(logits))

    def train_one(self, image, label):
        logits = self.forward(image)
        loss = self.loss.forward(logits, label)

        gradient = self.loss.backward()
        gradient = self.dense.backward(gradient)
        gradient = self.flatten.backward(gradient)
        gradient = self.pool.backward(gradient)
        gradient = self.relu.backward(gradient)
        self.conv.backward(gradient)

        return loss

    def evaluate(self, images, labels):
        predictions = [self.predict(image) for image in images]
        return np.mean(np.array(predictions) == labels)
