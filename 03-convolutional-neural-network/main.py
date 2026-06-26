"""
main.py
~~~~~~~

Train a small CNN from scratch on simple 8x8 image patterns.

Class 0: vertical line
Class 1: horizontal line
Class 2: diagonal line
"""

import numpy as np

from cnn import TinyCNN


def make_pattern(label, rng):
    image = rng.normal(0.0, 0.08, (8, 8))

    if label == 0:
        col = rng.integers(2, 6)
        image[:, col] += 1.0
    elif label == 1:
        row = rng.integers(2, 6)
        image[row, :] += 1.0
    else:
        if rng.random() < 0.5:
            np.fill_diagonal(image, image.diagonal() + 1.0)
        else:
            np.fill_diagonal(np.fliplr(image), np.fliplr(image).diagonal() + 1.0)

    return np.clip(image, 0.0, 1.0)


def make_dataset(samples_per_class=80, seed=7):
    rng = np.random.default_rng(seed)
    images = []
    labels = []

    for label in range(3):
        for _ in range(samples_per_class):
            images.append(make_pattern(label, rng))
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    order = rng.permutation(len(labels))
    return images[order], labels[order]


def train():
    images, labels = make_dataset()
    split = int(len(labels) * 0.8)
    train_images, test_images = images[:split], images[split:]
    train_labels, test_labels = labels[:split], labels[split:]

    network = TinyCNN(learning_rate=0.01)

    print("Training a CNN from scratch")
    print(f"Training images: {len(train_images)}")
    print(f"Test images: {len(test_images)}")
    print()

    for epoch in range(12):
        total_loss = 0.0
        order = np.random.default_rng(epoch).permutation(len(train_labels))

        for index in order:
            total_loss += network.train_one(train_images[index], train_labels[index])

        accuracy = network.evaluate(test_images, test_labels)
        average_loss = total_loss / len(train_labels)
        print(f"Epoch {epoch + 1:02d} | loss {average_loss:.4f} | accuracy {accuracy:.2%}")

    print()
    print("Example predictions:")
    names = ["vertical", "horizontal", "diagonal"]
    for image, label in zip(test_images[:6], test_labels[:6]):
        prediction = network.predict(image)
        print(f"expected {names[label]:10s} -> predicted {names[prediction]}")


if __name__ == "__main__":
    train()
