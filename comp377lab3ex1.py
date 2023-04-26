# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 21:32:43 2023

@author: rafid
"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the student dataset
df = pd.read_csv('C:\AIForSoftDevelopers\student.cleaned.data.csv')

# Convert the target column to binary (0 if <10, 1 otherwise)
df['G3'] = (df['G3'] >= 10).astype(int)

# Select the features and target
features = ['traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'health']
target = 'G3'

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train the MLPClassifier model
clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1)
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy of the model
print(f"Accuracy: {accuracy * 100:.2f}%")



