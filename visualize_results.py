import numpy as np  # For loading predictions
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For confusion matrices
from sklearn.metrics import confusion_matrix  # For computing confusion matrices
import os  # For handling directories

# Define output directory
output_dir = './outputs'
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn’t exist

# Function to plot and save confusion matrix
def plot_confusion_matrix(labels, preds, title, filename):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Digit')
    plt.ylabel('True Digit')
    plt.title(title)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved confusion matrix as '{os.path.join(output_dir, filename)}'")

# Load and plot PyTorch results
try:
    pytorch_labels = np.load(os.path.join(output_dir, 'pytorch_labels.npy'))
    pytorch_preds = np.load(os.path.join(output_dir, 'pytorch_preds.npy'))
    plot_confusion_matrix(pytorch_labels, pytorch_preds,
                          'Confusion Matrix for MNIST (PyTorch)',
                          'confusion_matrix_pytorch.png')
except FileNotFoundError:
    print("PyTorch predictions or labels not found. Run mnist_pytorch.py first.")

# Load and plot TensorFlow results
try:
    tensorflow_labels = np.load(os.path.join(output_dir, 'tensorflow_labels.npy'))
    tensorflow_preds = np.load(os.path.join(output_dir, 'tensorflow_preds.npy'))
    plot_confusion_matrix(tensorflow_labels, tensorflow_preds,
                          'Confusion Matrix for MNIST (TensorFlow)',
                          'confusion_matrix_tensorflow.png')
except FileNotFoundError:
    print("TensorFlow predictions or labels not found. Run mnist_tensorflow.py first.")
