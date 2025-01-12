from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # Output layer for 10 classes in CIFAR-10
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# Train the model on CIFAR-10 dataset
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))


# Evaluate the model on CIFAR-10 test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy on CIFAR-10: {test_acc}")


# Load CIFAR-100 dataset
(x_train_100, y_train_100), (x_test_100, y_test_100) = cifar100.load_data()

# Normalize the images to [0, 1]
x_train_100, x_test_100 = x_train_100 / 255.0, x_test_100 / 255.0

# One-hot encode the labels for CIFAR-100
y_train_100 = to_categorical(y_train_100, 100)
y_test_100 = to_categorical(y_test_100, 100)


# Modify the final layer for CIFAR-100
model.pop()  # Remove the last layer (for CIFAR-10)

# Add the new output layer for CIFAR-100 with 100 classes
model.add(layers.Dense(100, activation='softmax'))

# Recompile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Display the updated model summary
model.summary()



# Retrain the model on CIFAR-100 dataset
history_100 = model.fit(x_train_100, y_train_100, epochs=10, batch_size=64, validation_data=(x_test_100, y_test_100))

# Evaluate the model on CIFAR-100 test data
test_loss_100, test_acc_100 = model.evaluate(x_test_100, y_test_100, verbose=2)
print(f"Test accuracy on CIFAR-100: {test_acc_100}")


# Plotting training history
plt.plot(history.history['accuracy'], label='CIFAR-10 Train Accuracy')
plt.plot(history.history['val_accuracy'], label='CIFAR-10 Validation Accuracy')
plt.plot(history_100.history['accuracy'], label='CIFAR-100 Train Accuracy')
plt.plot(history_100.history['val_accuracy'], label='CIFAR-100 Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
