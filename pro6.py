import tensorflow as tf
from tensorflow.keras import layers, models, datasets, utils
from sklearn.metrics import classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np


### Load and Preprocess MNIST Dataset

def load_mnist():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    # Normalize and reshape for compatibility with LeNet-5
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension
    x_test = np.expand_dims(x_test, axis=-1)
    y_train = utils.to_categorical(y_train, 10)  # One-hot encode labels
    y_test = utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


### Define the LeNet-5 Architecture
def build_lenet5():
    model = models.Sequential([
        layers.Conv2D(6, kernel_size=5, activation='tanh', input_shape=(28, 28, 1), padding='same'),
        layers.AvgPool2D(pool_size=2, strides=2),
        layers.Conv2D(16, kernel_size=5, activation='tanh', padding='valid'),
        layers.AvgPool2D(pool_size=2, strides=2),
        layers.Flatten(),
        layers.Dense(120, activation='tanh'),
        layers.Dense(84, activation='tanh'),
        layers.Dense(10, activation='softmax')
    ])
    return model


### Plot Training/Validation Metrics

def plot_metrics(history):
    # Accuracy plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    ### Train and Evaluate the Model

    (x_train, y_train), (x_test, y_test) = load_mnist()

# Build the model
model = build_lenet5()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, validation_split=0.1, batch_size=64, epochs=10, verbose=1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Generate classification report
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Classification metrics
precision, recall, f1, _ = precision_recall_fscore_support(y_true_classes, y_pred_classes, average='macro')
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=[str(i) for i in range(10)]))

# Plot training/validation metrics
plot_metrics(history)