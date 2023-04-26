# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 23:05:42 2023

@author: rafid
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X, y = fetch_california_housing(return_X_y = True , as_frame = True)
X.head()
type(y)
y.head()
X = X[['AveRooms',	'AveBedrms']]
X.head()

transformer = PolynomialFeatures(degree = 2, include_bias=False)
transformer.fit(X)
X_new = transformer.transform(X)
print(X_new)
X_new.shape


model = LinearRegression()
model.fit(X_new , y)
r_sq = model.score(X_new, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('coefficients:', model.coef_)

y_pred = model.predict( X_new )
y_pred