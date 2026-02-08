# Neural Network From Scratch

**Part 2 & 3 of "AI From First Principles"**

A 3-layer neural network that learns to recognize handwritten digits with **96% accuracy** in just 74 lines of Python.

## Quick Start
```bash
cd 02-neural-network
pip install numpy
python download_mnist.py
python main.py
```

## What You'll Build

Starting with random weights → 96% accuracy after 30 epochs.

**Architecture:**
- 784 input neurons (28×28 pixels)
- 30 hidden neurons
- 10 output neurons (digits 0-9)

## Files

- `network.py` - The neural network implementation
- `mnist_loader.py` - Loads MNIST data
- `main.py` - Training script
- `download_mnist.py` - Downloads dataset

## How It Works

1. **Forward pass**: Make prediction
2. **Measure error**: How wrong?
3. **Backpropagation**: Calculate gradients
4. **Update weights**: Reduce error
5. **Repeat** 1.5 million times

Result: Intelligence emerges from organized trial and error.

## Read More

Full explanation: [Part 3 on Medium](https://medium.com/@ramadhan.zome)