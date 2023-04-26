# -*- coding: utf-8 -*-
"""
@author: rafid

"""

import tensorflow as tf
from sklearn.neural_network import MLPClassifier

# Fetch the MNIST dataset using tensorflow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data by reshaping and normalizing
x_train = x_train.reshape(60000, 784) / 255.0
x_test = x_test.reshape(10000, 784) / 255.0

# Train the MLPClassifier model using the top 60 rows of the training data
clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1)
clf.fit(x_train[:60], y_train[:60])

# Test the model using the top 10 rows of the test data
accuracy = clf.score(x_test[:10], y_test[:10])

# Print the accuracy of the model
print(f"Accuracy: {accuracy * 100:.2f}%")

