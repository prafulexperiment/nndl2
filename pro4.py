import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


def load_cifar(data_path, is_cifar100=False):
    def load_file(file):
        with open(f"{data_path}/{file}", 'rb') as f:
            return pickle.load(f, encoding='bytes')
    
    if is_cifar100:
        train, test = load_file('train'), load_file('test')
        x_train, y_train = train[b'data'], np.array(train[b'fine_labels'])
        x_test, y_test = test[b'data'], np.array(test[b'fine_labels'])
    else:
        x_train, y_train = [], []
        for i in range(1, 6):
            batch = load_file(f"data_batch_{i}")
            x_train.append(batch[b'data'])
            y_train += batch[b'labels']
        x_train, y_train = np.concatenate(x_train), np.array(y_train)
        test = load_file("test_batch")
        x_test, y_test = test[b'data'], np.array(test[b'labels'])
    
 
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) / 255.0
    return (x_train, y_train), (x_test, y_test)


def create_cnn(num_classes):
    return models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])


def train_model(x_train, y_train, x_test, y_test, num_classes, dataset_name):
    print(f"Training on {dataset_name}")
    model = create_cnn(num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f"{dataset_name} Training History")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{dataset_name} Test Accuracy: {test_acc:.2f}\n")


(x_train_10, y_train_10), (x_test_10, y_test_10) = load_cifar('cifar-10-batches-py')
(x_train_100, y_train_100), (x_test_100, y_test_100) = load_cifar('cifar-100-python', is_cifar100=True)


y_train_10 = tf.keras.utils.to_categorical(y_train_10, 10)
y_test_10 = tf.keras.utils.to_categorical(y_test_10, 10)
y_train_100 = tf.keras.utils.to_categorical(y_train_100, 100)
y_test_100 = tf.keras.utils.to_categorical(y_test_100, 100)


train_model(x_train_10, y_train_10, x_test_10, y_test_10, 10, "CIFAR-10")
train_model(x_train_100, y_train_100, x_test_100, y_test_100, 100, "CIFAR-100")
