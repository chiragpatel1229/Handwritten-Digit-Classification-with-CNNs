# Handwritten Digit Classification with CNNs

## Project Overview
This project demonstrates the implementation of Convolutional Neural Networks (CNNs) to classify handwritten digits (0–9) from the MNIST dataset using two popular deep learning frameworks: **PyTorch** and **TensorFlow**. This machine learning project demonstrates fundamental skills in data preprocessing, model building, training, evaluation, and visualization. The MNIST dataset is stored locally for efficient processing, and the project is structured into separate scripts for modularity and clarity.

**Key Features**:
- Implements CNNs in both PyTorch and TensorFlow, achieving ~97–98% test accuracy.
- Downloads and stores the MNIST dataset locally in `./mnist_data` to avoid redundant downloads.
- Generates confusion matrices to visualize model performance using Matplotlib and Seaborn.
- Organized into distinct scripts for data downloading, training, and visualization.

**Tech Stack**:
- Machine Learning (Supervised Learning, CNNs)
- Python, PyTorch, TensorFlow
- Data Preprocessing and Local Storage
- Model Evaluation and Visualization
- Project Organization and Documentation

**Completion Date**: June 2025

### File Descriptions
- **`download_mnist.py`**: Downloads the MNIST dataset using both PyTorch and TensorFlow, storing it in `./mnist_data` for fast access in training scripts.
- **`mnist_pytorch.py`**: Loads MNIST from `./mnist_data`, trains a simple CNN using PyTorch, evaluates performance, and saves predictions to `./outputs`.
- **`mnist_tensorflow.py`**: Loads MNIST from `./mnist_data`, trains a simple CNN using TensorFlow, evaluates performance, and saves predictions to `./outputs`.
- **`visualize_results.py`**: Loads predictions from `./outputs` and generates confusion matrices, saved as `.png` files in `./outputs`.
- **`mnist_data/`**: Contains MNIST dataset files (e.g., `mnist.npz` for TensorFlow, raw/processed files for PyTorch).
- **`outputs/`**: Stores model predictions (`.npy` files) and confusion matrices (`.png` files).

## Project Details
### Dataset
- **MNIST**: A dataset of 60,000 training and 10,000 test images (28x28 grayscale) of handwritten digits (0–9).
- **Storage**: Saved in `./mnist_data` for fast access, avoiding repeated downloads.
- **Preprocessing**: Normalized pixel values to [-1, 1] for PyTorch, [0, 1] for TensorFlow.

### Model Architecture
- **CNN** (both PyTorch and TensorFlow):
  - 2 convolutional layers (16 and 32 filters, 3x3 kernels, ReLU activation).
  - 2 max-pooling layers (2x2).
  - 2 fully connected layers (128 units, 10 output units).
  - Dropout (25%) to prevent overfitting.
- **Training**: 3 epochs, batch size 64, Adam optimizer, cross-entropy loss.
- **Accuracy**: ~97–98% on test data.

### Visualization
- **Confusion Matrices**: Show prediction accuracy for each digit, saved as PNGs in `./outputs`.
- **Tools**: Matplotlib and Seaborn for clear, professional visualizations.

## Future Improvements
- Add data augmentation (e.g., rotation, scaling) to improve accuracy.
- Experiment with deeper CNN architectures or different datasets (e.g., Fashion-MNIST).
- Save and load trained models for reuse.
- Add hyperparameter tuning (e.g., learning rate, number of filters).

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---
*Last updated: June 18, 2025*
