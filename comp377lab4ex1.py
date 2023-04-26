# -*- coding: utf-8 -*-
"""

@author: rafid

"""

import tensorflow as tf

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape the input data to be in the format expected by Conv2D layers
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.AveragePooling2D(),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.AveragePooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

# Compile the model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

# Train the model using the top 60 rows out of 60000 rows of the training data
model.fit(x_train[:60], y_train[:60], epochs=10, validation_split=0.2)

# Test the model using top 10 rows out of 10000 rows of test data
test_loss, test_acc = model.evaluate(x_test[:10], y_test[:10])
print('Test accuracy:', test_acc)
