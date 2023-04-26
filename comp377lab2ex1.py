# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:29:26 2023

@author: rafid
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('C:\AIForSoftDevelopers\student.cleaned.data.csv')

# Convert the target column G3 to binary values
data['G3'] = data['G3'].apply(lambda x: 1 if x >= 10 else 0)

# Select the features and target
features = ['traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'health']
target = 'G3'
X = data[features]
y = data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy}")


