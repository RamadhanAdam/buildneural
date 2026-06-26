# Convolutional Neural Network From Scratch

**Part 4 of "AI From First Principles"**

A small convolutional neural network built with only NumPy. It learns to classify tiny images by looking for simple visual patterns: vertical lines, horizontal lines, and diagonal lines.

The goal is not to beat modern computer vision models. The goal is to see what a CNN is actually doing.

## Quick Start

```bash
cd 03-convolutional-neural-network
pip install -r requirements.txt
python main.py
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

## Files

- `layers.py` - Convolution, ReLU, max pooling, flatten, dense, and loss layers
- `cnn.py` - Wires the layers into a tiny CNN
- `main.py` - Generates simple image data and trains the network
- `requirements.txt` - Minimal dependency list

## How It Works

1. **Convolution**: Small filters slide across the image and look for local patterns.
2. **ReLU**: Negative signals are removed so useful activations stand out.
3. **Max pooling**: The strongest signal in each small region is kept.
4. **Flattening**: The image-like feature maps become one long vector.
5. **Dense layer**: The final classifier turns features into class scores.
6. **Backpropagation**: Every layer receives a correction signal and updates itself.

This is the core idea behind CNNs: learn small reusable filters, then combine them into higher-level visual understanding.
