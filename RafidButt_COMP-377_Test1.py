# -*- coding: utf-8 -*-
"""
@author: rafid

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from fractions import Fraction

# Load the data
df = pd.read_csv('C:\AIForSoftDevelopers\heart.csv')

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

# Manually set the test set to be the bottom 120 rows of the original dataframe
test_df = df.tail(120)

# Create the feature and target arrays
X_train = train_df.drop('disease', axis=1)
y_train = train_df['disease']
X_test = test_df.drop('disease', axis=1)
y_test = test_df['disease']

# Build a logistic regression classifier model
log_reg = LogisticRegression()

# Train the logistic regression model
log_reg.fit(X_train, y_train)

# Test the logistic regression model
y_pred = log_reg.predict(X_test)

# Display the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')

# Print the Classification Report
print(classification_report(y_test, y_pred))

# Print the number of correctly classified samples
num_correct = accuracy_score(y_test, y_pred, normalize=False)
print("Number of correctly classified samples:", num_correct)

# Print the fraction of correctly classified samples as a simplified fraction and decimal
fraction = Fraction(num_correct, len(y_test))
decimal = round(accuracy_score(y_test, y_pred), 2)
print("Fraction of correctly classified samples: {} = {}".format(fraction, decimal))


# Answer to Q9: The sum of the numbers in a single row of the Confusion Matrix signifies the total number of samples from that class, regardless of whether they were classified correctly or incorrectly.
