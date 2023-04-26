# -*- coding: utf-8 -*-
"""
@author: rafid
"""



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
 


data = pd.read_excel('C:\AIForSoftDevelopers\Real estate valuation data set.xlsx')
 


x = data.iloc[:,[2, 3, 4, 5, 6]]
y = data[["Y house price of unit area"]]

 


train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)

 


linear_reg = LinearRegression()
linear_reg.fit(train_x, train_y)
 


pred_y = linear_reg.predict(test_x)
 


print("R2 score =", r2_score(test_y, pred_y))
