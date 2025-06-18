# Script to download and store the MNIST dataset locally
# This ensures fast access for training scripts by avoiding repeated downloads

import os  # For handling directories
import torchvision  # For accessing MNIST in PyTorch
import torchvision.transforms as transforms  # For preprocessing
import tensorflow as tf  # For accessing MNIST in TensorFlow

# Define directory to store MNIST dataset
data_dir = './mnist_data'
os.makedirs(data_dir, exist_ok=True)  # Create directory if it doesn’t exist

# Download MNIST using PyTorch (saves raw and processed files)
print("Downloading MNIST for PyTorch...")
transform = transforms.Compose([transforms.ToTensor()])
# Download training and test data
trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
print(f"PyTorch MNIST data saved to {data_dir}")

# Download MNIST using TensorFlow (saves mnist.npz file)
print("Downloading MNIST for TensorFlow...")
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(path=os.path.join(data_dir, 'mnist.npz'))
print(f"TensorFlow MNIST data saved to {data_dir}")

print("MNIST dataset download complete. Ready for use in training scripts.")