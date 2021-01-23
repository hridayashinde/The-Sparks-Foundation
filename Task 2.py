#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 09:42:34 2021

@author: hridayashinde
"""

# Importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Loading the iris dataset
iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
# The first 5 rows
print(iris_df.head())

#Using k-means model for different values of k and computing Sum of Squared Differences
Sum_of_squared_distances = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(iris_df)
    Sum_of_squared_distances.append(km.inertia_)
    
    
#Plotting Sum of Squared Differences for various k values    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

print('In the plot above the elbow is at k=4 indicating the optimal k for this dataset is 4')