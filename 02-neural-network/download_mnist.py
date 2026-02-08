"""
download_mnist.py
~~~~~~~~~~~~~~~~~

Downloads the MNIST dataset and saves it in the required format.
"""

import os
import urllib.request


def download_mnist():
    """Download MNIST dataset from the official source."""
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created 'data' directory")
    
    # Check if already downloaded
    if os.path.exists('data/mnist.pkl.gz'):
        print("MNIST dataset already exists at 'data/mnist.pkl.gz'")
        print("Delete it if you want to re-download")
        return
    
    print("Downloading MNIST dataset...")
    print("This may take a minute (~15 MB)")
    
    url = 'https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz'
    
    try:
        urllib.request.urlretrieve(url, './data/mnist.pkl.gz')
        print("\n✓ Download complete!")
        print("Dataset saved to: data/mnist.pkl.gz")
        print("\nYou can now run: python main.py")
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("\nAlternative: Download manually from:")
        print(url)


if __name__ == "__main__":
    download_mnist()
