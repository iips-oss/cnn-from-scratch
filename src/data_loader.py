import os
import numpy as np
import urllib.request
import gzip


def fetch_mnist(data_dir="data"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        'x_train': 'train-images-idx3-ubyte.gz',
        'y_train': 'train-labels-idx1-ubyte.gz',
        'x_test': 't10k-images-idx3-ubyte.gz',
        'y_test': 't10k-labels-idx1-ubyte.gz'
    }

    data = {}
    # storing the data
    for name, filename in files.items():
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            print("Downloading {}".format(filename))
            urllib.request.urlretrieve(base_url + filename, filepath)

        with gzip.open(filepath, 'rb') as f:
            if 'images' in filename:
                data[name] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
            else:
                data[name] = np.frombuffer(f.read(), np.uint8, offset=8)

    # normalize
    x_train = (data['x_train'] / 255.0) - 0.5
    x_test = (data['x_test'] / 255.0) - 0.5

    y_train = data['y_train']
    y_test = data['y_test']

    return x_train, y_train, x_test, y_test


# for testing purpose
if __name__ == "__main__":
    print("Loading MNIST data...")
    x_train, y_train, x_test, y_test = fetch_mnist(data_dir="../data")
    print(f"Training images: {x_train.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Pixel range: Min = {np.min(x_train[0])}, Max = {np.max(x_train[0])}")
