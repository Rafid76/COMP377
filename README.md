# COMP377
# AI For Software Developers - Labs

Lab Assignment #1 – Apply Linear Regression and Polynomial Regression algorithms to solve various regression problems

Exercise 1: Linear Regression

Write an application using scikit-learn to train/test the real estate data. Use Linear Regression model. Use the dataset from UCI repository: https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set.

The target in the dataset is “Y house price of unit area”. Also, determine the coefficient of determination (R2) of the model.


Exercise 2: Polynomial Regression

Write an application using scikit-learn to train/test the real estate data. Use Polynomial Regression model. The dataset is from California housing data: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html. Use only the two features AveRooms and AveBedrms (out of the eight features). You should invoke the PolynomialFeatures constructor as PolynomialFeatures(degree=2, include_bias=False). 
(To understand this way of calling the PolynomialFeatures constructor, the website https://realpython.com/linear-regression-in-python/#polynomial-regression-with-scikit-learn may be helpful.)

Also, determine the coefficient of determination (R2) of the model.

(Hint: X, y = fetch_california_housing(return_X_y=True, as_frame=True) can be used to return the feature data in X (in form of a DataFrame object) and target data in y. Check the type of y. If the type of y is found to be DataFrame, then convert the type of y to Series using the DataFrame method squeeze.)





Lab Assignment #2 – Apply Logistic Regression and Support Vector Machines algorithms to solve various classification problems

Exercise 1: Logistic Regression

Write a scikit-learn based application to predict the secondary school student performance using a logistic regression model. The dataset is present in file student.cleaned.data.csv. The features to be taken into account are traveltime, studytime, failures, famrel, freetime, gout, health. The target should be G3. In G3 column, assume the values less than 10 to be 0, and the values equal to or more than 10 to be 1. Evaluate the accuracy of the model.


Exercise 2: Support Vector Machines

Write a scikit-learn based application to classify MNIST digits using a Support Vectors Machine (SVM) model. The dataset is from http://yann.lecun.com/exdb/mnist/. You must use a tensorflow function to just fetch the data. The description about this tensorflow function is in this page: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data
Rest of the functionality must be accomplished using scikit-learn library. Train the model using the top 60 rows out of 60000 rows of the training data (present in x_train; see below how to obtain the training data in x_train). Test the model using top 10 rows out of 10000 rows of test data (present in x_test; see below how to obtain the test data in x_test). Evaluate the accuracy of the model.





Lab Assignment #3 – Use MLPs for prediction/classification problems

Exercise 1: Multi-layer Neural Network

Write a scikit-learn based application to predict the secondary school student performance using the MLP model MLPClassifier. The dataset is present in file student.cleaned.data.csv. The features to be taken into account are traveltime, studytime, failures, famrel, freetime, gout, health. The target should be G3. In G3 column, assume the values less than 10 to be 0, and the values equal to or more than 10 to be 1. Evaluate the accuracy of the model. 


Exercise 2: Multi-layer Neural Network

Write a scikit-learn based application to classify MNIST digits using the MLP model MLPClassifier. The dataset is from http://yann.lecun.com/exdb/mnist/. You must use a tensorflow function to just fetch the data. The description about this tensorflow function is in this page: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data
Rest of the functionality must be accomplished using scikit-learn library. Train the model using the top 60 rows out of 60000 rows of the training data (present in x_train; see below how to obtain the training data in x_train). Test the model using top 10 rows out of 10000 rows of test data (present in x_test; see below how to obtain the test data in x_test). Evaluate the accuracy of the model. 





Lab Assignment #4 – Use CNNs in image classification problems

Exercise 1: Convolutional Neural Networks

In this exercise you will implement a CNN model for digit classification using tensorflow and MNIST dataset. The site https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data will help you to figure out how to fetch the data using a special tensorflow function. Overall, use convolution layers and pooling layers that are eventually followed by a densely connected output layer. Here is the architecture of the network:
-	Use Conv2D for two hidden layers. They are activated by ReLU activation function
-	After each Conv2D layer, use a mean pooling operation. 
-	Use Flatten operation.
-	Use Dense for the output layer. This layer is to be activated by softmax function.
-	Do not use Dropout operation in this architecture.
Do not use one-hot encoding of the target.
Compile the model using the loss function SparseCategoricalCrossentropy (You need to use this loss function since you must not use one-hot encoding of the target). Train the model using the top 60 rows out of 60000 rows of the training data (present in x_train; see below how to obtain the training data in x_train). Test the model using top 10 rows out of 10000 rows of test data (present in x_test; see below how to obtain the test data in x_test). Evaluate the accuracy of the model.


Exercise 2: Convolutional Neural Networks

In this exercise you will build a CNN model for photo classification using tensorflow and CIFAR-10 dataset. The site  https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data  will help you to figure out how to fetch the data using a special tensorflow function. Use the same technique as in Exercise 1 above to create the CNN architecture. Train the model using the top 50 rows out of 50000 rows of the training data (present in x_train; see below how to obtain the training data in x_train). Test the model using top 10 rows out of 10000 rows of test data (present in x_test; see below how to obtain the test data in x_test). Evaluate the accuracy of the model. 




Lab Assignment #5 – Apply LSTM algorithm to make future predictions using time series data

Exercise 1: LSTM

In this exercise you will implement an LSTM model to make future predictions using time series data. Use TensorFlow to build an LSTM model for predicting stock prices for a company listed in the NASDAQ listings. For this assignment, you should first download the historic data of a company’s stock price in form of a .csv file. Then, use the data displayed in the column named Close. This column contains the closing price (i.e. the last price) of the day of a stock.

Evaluate the model test loss. Display the graph of real data and predicted data.




Test #1

Exercise 1: 
In this exercise you will use logistic regression to train a model for predicting the diagnosis of heart disease using the heart disease dataset from the file heart.csv. The detail about the column headers is provided in the readme.txt file. You are to code your solution in a Jupyter notebook file (i.e. an ipynb file). This exercise tests your familiarity with scikit-learn (sklearn) classes/functions as well.

You should ONLY use the following import statements:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

You must use appropriate sklearn classes/functions to answer the following questions. For display, you should ONLY use the plt specified in one of the import statements above. DO NOT define your own classes/functions.

1.	Use the only appropriate sklearn function, split the data present in heart.csv for training and predicting, 
a.	without shuffling, in such a way that  
b.	the bottom 120 rows of the data in heart.csv always form the set of examples to be used for prediction every time your program is run. 
2.	Build a logistic regression classifier model. 
3.	Train the above model to obtain a parameterized model. 
4.	Test using that parameterized model.  
5.	Display the Confusion Matrix such that the cells of the matrix are colored.  
6.	Print the Classification Report. 
7.	Print the number of correctly classified samples using a specific scikit-learn function.     
8.	Print the fraction of correctly classified samples using a specific scikit-learn function.             
9.	Given a set of classes, when you generate a Confusion Matrix using a given row of true target data and a given column of corresponding predicted target data, what does the sum of the numbers in a single row of the Confusion Matrix signify? Write the answer to this question as a comment at the end/bottom of your Jupyter notebook file (the .ipynb file).

