#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 10:40:54 2021

@author: hridayashinde
"""

import numpy as np
import pandas as pd
import seaborn as sb


# Importing the data

data = pd.read_csv('/Users/hridayashinde/Desktop/Sparks Foundation/Dataset1.csv')
X = data['Hours']
y = data['Scores']

# Plotting the Data

sb.scatterplot(X,y)
sb.lmplot('Hours','Scores',data)

# Splitting the data into training and testing data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train = np.array(X_train).reshape(-1,1)

# Fitting a Linear Regression model

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)

print('Regression Coefficients' , reg.coef_)
print('Regression Intercept:' , reg.intercept_)
print('Regression Score on training data = ',reg.score(X_train,y_train))
print('Regression Score on testing data = ',reg.score(np.array(X_test).reshape(-1,1),y_test))

#Predicting the Sccore for Hours = 9.25

print('Score if student syudies for 9.25 hrs/day:',reg.predict(np.array(9.25).reshape(-1,1)))