import tensorflow as tf  # TensorFlow for building the neural network
import numpy as np  # For saving predictions
import os  # For handling directories

# Define directories for data and outputs
data_dir = './mnist_data'  # Folder where MNIST dataset is stored
output_dir = './outputs'  # Folder to store predictions and visualizations
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn’t exist

# Check if MNIST data exists
mnist_file = os.path.join(data_dir, 'mnist.npz')
if not os.path.exists(mnist_file):
    raise FileNotFoundError(f"MNIST data not found in {mnist_file}. Run download_mnist.py first.")

# Load MNIST dataset from local folder
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(path=mnist_file)

# Normalize and reshape data
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
test_images = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Define a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model for 3 epochs
print("Starting training...")
model.fit(train_images, train_labels, batch_size=64, epochs=3, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Generate predictions
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Save predictions and labels to output_dir
np.save(os.path.join(output_dir, 'tensorflow_preds.npy'), predicted_labels)
np.save(os.path.join(output_dir, 'tensorflow_labels.npy'), test_labels)
print(f"Predictions and labels saved to {output_dir}")
