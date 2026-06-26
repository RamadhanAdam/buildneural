# Convolutional Neural Network From Scratch

**Part 4 of "AI From First Principles"**

A small convolutional neural network built with only NumPy. It learns to classify tiny 8x8 images by looking for simple visual patterns: vertical lines, horizontal lines, and diagonal lines.

The goal is not to beat modern computer vision models. The goal is to see what a CNN is actually doing.

## Quick Start

```bash
cd 03-convolutional-neural-network
pip install -r requirements.txt
python main.py
```

Expected output will look similar to this:

```text
Training a CNN from scratch
Training images: 192
Test images: 48

Epoch 01 | loss 0.9950 | accuracy 54.17%
Epoch 02 | loss 0.7161 | accuracy 89.58%
Epoch 03 | loss 0.5019 | accuracy 97.92%
...
Epoch 12 | loss 0.0229 | accuracy 100.00%
```

## What You'll Build

**Architecture:**

- 8x8 grayscale image input
- 3x3 convolution filters
- ReLU activation
- 2x2 max pooling
- Flatten layer
- Dense classifier
- Softmax cross-entropy loss

```text
Input image
8 x 8 x 1
    |
    v
Convolution
4 filters, each 3 x 3
Output: 6 x 6 x 4
    |
    v
ReLU
Output: 6 x 6 x 4
    |
    v
Max Pooling
2 x 2 windows
Output: 3 x 3 x 4
    |
    v
Flatten
Output: 36 numbers
    |
    v
Dense Layer
Output: 3 class scores
    |
    v
Softmax + Cross-Entropy
Prediction + loss
```

## Files

- `layers.py` - Convolution, ReLU, max pooling, flatten, dense, and loss layers
- `cnn.py` - Wires the layers into a tiny CNN
- `main.py` - Generates simple image data and trains the network
- `requirements.txt` - Minimal dependency list

## Dataset

This lesson does not download MNIST. Instead, `main.py` creates a tiny dataset in code so the whole example stays easy to inspect.

Each image is an 8x8 grid of pixel values between 0 and 1.

```text
Vertical class        Horizontal class      Diagonal class

. . . # . . . .       . . . . . . . .       # . . . . . . .
. . . # . . . .       . . . . . . . .       . # . . . . . .
. . . # . . . .       # # # # # # # #       . . # . . . . .
. . . # . . . .       . . . . . . . .       . . . # . . . .
. . . # . . . .       . . . . . . . .       . . . . # . . .
. . . # . . . .       . . . . . . . .       . . . . . # . .
. . . # . . . .       . . . . . . . .       . . . . . . # .
. . . # . . . .       . . . . . . . .       . . . . . . . #
```

Small random noise is added so the network cannot solve the task by memorizing one perfect image.

## How It Works

1. **Convolution**: Small filters slide across the image and look for local patterns.
2. **ReLU**: Negative signals are removed so useful activations stand out.
3. **Max pooling**: The strongest signal in each small region is kept.
4. **Flattening**: The image-like feature maps become one long vector.
5. **Dense layer**: The final classifier turns features into class scores.
6. **Backpropagation**: Every layer receives a correction signal and updates itself.

This is the core idea behind CNNs: learn small reusable filters, then combine them into higher-level visual understanding.

## Shape Walkthrough

The input image is 8x8. The convolution uses 3x3 filters with no padding, so a filter can only start in 6 positions across and 6 positions down.

```text
8 - 3 + 1 = 6
```

Because the network has 4 filters, convolution produces 4 separate 6x6 feature maps:

```text
6 x 6 x 4
```

Max pooling uses 2x2 windows, so each 6x6 feature map becomes 3x3:

```text
3 x 3 x 4 = 36 values
```

Those 36 values go into the dense layer, which produces 3 scores:

- score for vertical
- score for horizontal
- score for diagonal

The highest score becomes the prediction.

## Why This Is Still From Scratch

This project does not use TensorFlow, PyTorch, Keras, or scikit-learn.

The only dependency is NumPy. NumPy gives us arrays and math operations, but the CNN logic is written manually:

- filter sliding
- convolution output
- ReLU masking
- max pooling
- dense layer update
- softmax cross-entropy
- backpropagation through each layer

That means you can open `layers.py` and follow the actual learning process line by line.

## Reading Order

If you are studying the code, read it in this order:

1. `main.py` - See the data and training loop.
2. `cnn.py` - See how the layers connect.
3. `layers.py` - See how each layer works internally.

That path starts with the big picture, then moves into the machinery.
