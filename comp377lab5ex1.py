# -*- coding: utf-8 -*-
"""
Created on Sat Apr 1 16:24:48 2023

@author: rafid
"""

# Step 1: Import necessary libraries and load the data from the CSV file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the data from the CSV file
data = pd.read_csv('C:\AIForSoftDevelopers\AAPL.csv')

# Use the data displayed in the column named Close
data = data[['Close']]

# Step 2: Preprocess the data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

train_size = int(len(data) * 0.7)
train_data = data[:train_size]
test_data = data[train_size:]

# Create time series sequences
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data)-seq_length-1):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)

seq_length = 50
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Step 3: Build the LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Step 4: Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Evaluate the test loss
test_loss = model.evaluate(X_test, y_test)
print("Test loss:", test_loss)

# Step 5: Make predictions on the test data
predictions = model.predict(X_test)

# Step 6: Display the graph of real data and predicted data
plt.plot(y_test, label='Real Data')
plt.plot(predictions, label='Predicted Data')
plt.legend()
plt.show()








