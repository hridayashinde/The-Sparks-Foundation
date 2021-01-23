#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 11:06:19 2021

@author: hridayashinde
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree

# Importing the data
data = pd.read_csv('/Users/hridayashinde/Documents/Sparks Foundation/Iris.csv')
print('DATA \n', data)

#CLEANING THE DATA
data.drop('Id', axis = 1, inplace=True)
y= data['Species']
classnames='Target Names',y.unique()


y.replace('Iris-setosa',0, inplace=True)
y.replace('Iris-versicolor',1,inplace=True)
y.replace('Iris-virginica',2,inplace=True)
X= data.drop('Species', axis = 1)
featurenames = X.columns


# Splitting the data into training and testing data

from sklearn.model_selection import train_test_split
X_train, X_test1, y_train, y_test1 = train_test_split(X, y, test_size=0.4,random_state=42)
X_test, X_validate, y_test, y_validate = train_test_split(X_test1, y_test1, test_size=0.5,random_state=40)

print(X_train.shape, y_train.shape,X_validate.shape,y_validate.shape,X_test.shape,y_test.shape)

# Fitting a Decision Tree model
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth = 3 ,random_state=0)

clf.fit(X_train, y_train)

# SCORES ON TRAINING AND TESTING DATA

print('Score on training data is ',clf.score(X_train, y_train))
print('Score on set 2  data is ',clf.score(X_validate, y_validate))
clf.predict(X_test)
print('Score on testing data is',clf.score(X_test, y_test))

#Visualising the Decision Tree
fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (2,2), dpi=300)
tree.plot_tree(clf,feature_names = fn, class_names=cn,filled = True);
