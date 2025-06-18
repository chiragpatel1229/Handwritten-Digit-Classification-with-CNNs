# Handwritten Digit Classification with CNNs

## Project Overview
This project demonstrates the implementation of Convolutional Neural Networks (CNNs) to classify handwritten digits (0–9) from the MNIST dataset using two popular deep learning frameworks: **PyTorch** and **TensorFlow**. Designed as a beginner-friendly machine learning project, it showcases fundamental skills in data preprocessing, model building, training, evaluation, and visualization. The MNIST dataset is stored locally for efficient processing, and the project is structured into separate scripts for modularity and clarity.

**Key Features**:
- Implements CNNs in both PyTorch and TensorFlow, achieving ~97–98% test accuracy.
- Downloads and stores the MNIST dataset locally in `./mnist_data` to avoid redundant downloads.
- Generates confusion matrices to visualize model performance using Matplotlib and Seaborn.
- Organized into distinct scripts for data downloading, training, and visualization.
- Includes a resume-ready project description highlighting beginner-level ML skills.

**Skills Demonstrated**:
- Machine Learning (Supervised Learning, CNNs)
- Python, PyTorch, TensorFlow
- Data Preprocessing and Local Storage
- Model Evaluation and Visualization
- Project Organization and Documentation

**Completion Date**: June 2025

## Project Structure
The repository is organized as follows:
```
├── mnist_data/                 # Folder storing the MNIST dataset
├── outputs/                    # Folder for predictions and confusion matrices
├── download_mnist.py           # Script to download and store MNIST dataset
├── mnist_pytorch.py            # PyTorch implementation for training and evaluation
├── mnist_tensorflow.py         # TensorFlow implementation for training and evaluation
├── visualize_results.py        # Script to generate confusion matrices
└── README.md                   # Project documentation (this file)
```

### File Descriptions
- **`download_mnist.py`**: Downloads the MNIST dataset using both PyTorch and TensorFlow, storing it in `./mnist_data` for fast access in training scripts.
- **`mnist_pytorch.py`**: Loads MNIST from `./mnist_data`, trains a simple CNN using PyTorch, evaluates performance, and saves predictions to `./outputs`.
- **`mnist_tensorflow.py`**: Loads MNIST from `./mnist_data`, trains a simple CNN using TensorFlow, evaluates performance, and saves predictions to `./outputs`.
- **`visualize_results.py`**: Loads predictions from `./outputs` and generates confusion matrices, saved as `.png` files in `./outputs`.
- **`resume_description.md`**: A concise project description for inclusion in a resume or portfolio.
- **`mnist_data/`**: Contains MNIST dataset files (e.g., `mnist.npz` for TensorFlow, raw/processed files for PyTorch).
- **`outputs/`**: Stores model predictions (`.npy` files) and confusion matrices (`.png` files).

## Prerequisites
To run this project, ensure you have the following installed:
- **Python**: Version 3.8 or higher
- **Libraries**:
  - PyTorch and torchvision
  - TensorFlow
  - NumPy
  - Matplotlib
  - Seaborn
  - Scikit-learn

## Usage
Follow these steps to run the project:

1. **Download MNIST Data** (if not already done):
   ```bash
   python download_mnist.py
   ```
   - Output: Confirms data is saved in `./mnist_data`.

2. **Train and Evaluate with PyTorch**:
   ```bash
   python mnist_pytorch.py
   ```
   - Loads MNIST from `./mnist_data`.
   - Trains a CNN for 3 epochs (~97–98% accuracy).
   - Saves predictions to `./outputs/pytorch_preds.npy` and `./outputs/pytorch_labels.npy`.
   - Output: Training loss per epoch and test accuracy.

3. **Train and Evaluate with TensorFlow**:
   ```bash
   python mnist_tensorflow.py
   ```
   - Loads MNIST from `./mnist_data/mnist.npz`.
   - Trains a CNN for 3 epochs (~97–98% accuracy).
   - Saves predictions to `./outputs/tensorflow_preds.npy` and `./outputs/tensorflow_labels.npy`.
   - Output: Training progress and test accuracy.

4. **Visualize Results**:
   ```bash
   python visualize_results.py
   ```
   - Loads predictions from `./outputs`.
   - Generates confusion matrices: `./outputs/confusion_matrix_pytorch.png` and `./outputs/confusion_matrix_tensorflow.png`.
   - Output: Confirms where matrices are saved or warns if predictions are missing.

5. **Use Resume Description**:
   - Copy the content of `resume_description.md` into your resume’s “Projects” section or portfolio.

### Example Outputs
- **Console Output (mnist_pytorch.py)**:
  ```
  Using device: cpu
  Starting training...
  Epoch 1/3, Loss: 0.2356
  Epoch 2/3, Loss: 0.0768
  Epoch 3/3, Loss: 0.0543
  Test Accuracy: 97.85%
  Predictions and labels saved to ./outputs
  ```
- **Confusion Matrix** (from `visualize_results.py`):
  - PNG files in `./outputs` showing model performance for each digit.

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

## Tips for Beginners
- **Run in Order**: Execute `download_mnist.py` before training scripts to ensure data is available.
- **Check Outputs**: Inspect `./outputs` for predictions and visualizations.
- **Experiment**: Try changing the batch size (e.g., 32) or epochs (e.g., 5) in `mnist_pytorch.py` or `mnist_tensorflow.py` to see their impact.
- **Learn Concepts**:
  - **Normalization**: Scales pixel values for better model performance.
  - **Convolution**: Detects patterns like edges in images.
  - **Dropout**: Reduces overfitting by ignoring some neurons during training.
- **Debugging**: If errors occur, verify library installations and ensure `./mnist_data` exists.
- **Jupyter Notebook**: Convert scripts to a notebook for step-by-step learning (ask for help if needed).

## Future Improvements
- Add data augmentation (e.g., rotation, scaling) to improve accuracy.
- Experiment with deeper CNN architectures or different datasets (e.g., Fashion-MNIST).
- Save and load trained models for reuse.
- Add hyperparameter tuning (e.g., learning rate, number of filters).

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact
For questions or feedback, feel free to open an issue in the repository or contact the project maintainer via GitHub.

---
*Last updated: June 18, 2025*
