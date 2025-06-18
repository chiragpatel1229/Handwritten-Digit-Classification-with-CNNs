# Handwritten Digit Classification using PyTorch
# Loads MNIST data from local folder and trains a simple CNN

# Import necessary libraries
import torch  # PyTorch for building and training the neural network
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimizers like Adam
import torchvision  # For accessing the MNIST dataset
import torchvision.transforms as transforms  # For data preprocessing
from torch.utils.data import DataLoader  # For batching data
import numpy as np  # For saving predictions
import os  # For handling directories

# Define directories for data and outputs
data_dir = './mnist_data'  # Folder where MNIST dataset is stored
output_dir = './outputs'  # Folder to store predictions and visualizations
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn’t exist

# Check if MNIST data exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"MNIST data not found in {data_dir}. Run download_mnist.py first.")

# Check if a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load and preprocess the MNIST dataset from data_dir
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1]
])

# Load training and test data (no download needed)
trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)

# Create data loaders
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Define a simple CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create the model and move it to the device
model = CNN().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model for 3 epochs
num_epochs = 3
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(trainloader):.4f}')

# Evaluate the model
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

# Save predictions and labels to output_dir
np.save(os.path.join(output_dir, 'pytorch_preds.npy'), all_preds)
np.save(os.path.join(output_dir, 'pytorch_labels.npy'), all_labels)
print(f"Predictions and labels saved to {output_dir}")