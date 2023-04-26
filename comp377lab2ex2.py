# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 16:04:55 2023

@author: rafid
"""

import numpy as np
from sklearn import svm
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset using TensorFlow's load_data function
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the training data into a 2D array
x_train = x_train.reshape(x_train.shape[0], -1)

# Take the top 60 rows out of 60000 for training
x_train = x_train[:60, :]
y_train = y_train[:60]

# Reshape the test data into a 2D array
x_test = x_test.reshape(x_test.shape[0], -1)

# Take the top 10 rows out of 10000 for testing
x_test = x_test[:10, :]
y_test = y_test[:10]

# Create an SVM classifier and fit it to the training data
clf = svm.SVC()
clf.fit(x_train, y_train)

# Use the classifier to predict the test data
y_pred = clf.predict(x_test)

# Evaluate the accuracy of the classifier
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)
