# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:43:51 2020

@author: rhaen
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Create dataset

# Class1
x1 = np.random.normal(25,5,100) # 25 ortalamaya ve 5 sigmaya sahip 1000 tane dğer üret
y1 = np.random.normal(25,5,100)

# Class2
x2 = np.random.normal(55,5,100) # 25 ortalamaya ve 5 sigmaya sahip 1000 tane dğer üret
y2 = np.random.normal(60,5,100)

# Class3
x3 = np.random.normal(55,5,100) # 25 ortalamaya ve 5 sigmaya sahip 1000 tane dğer üret
y3 = np.random.normal(15,5,100)

x = np.concatenate((x1, x2, x3),axis=0)
y = np.concatenate((y1, y2, y3),axis=0)

dictionary = {"x":x,"y":y}

data = pd.DataFrame(dictionary)

#%%
plt.scatter(x1,y1,color = "black")
plt.scatter(x2,y2,color = "black")
plt.scatter(x3,y3,color = "black")

#%% Dendogram

from scipy.cluster.hierarchy import linkage, dendrogram
merg = linkage(data,method="ward") # ward = bizim clusterlarımızın içindeki yayları minimize eder
dendrogram(merg,leaf_rotation = 90)
plt.xlabel("data points")
plt.ylabel("euclidean distance")

#%% HC
from sklearn.cluster import AgglomerativeClustering

hierartical_cluster = AgglomerativeClustering(n_clusters = 3, affinity = "euclidean",linkage = "ward")
cluster = hierartical_cluster.fit_predict(data)

data["label"] = cluster
plt.scatter(data.x[data.label == 0],data.y[data.label == 0 ],color = "red")
plt.scatter(data.x[data.label == 1],data.y[data.label == 1 ],color = "green")
plt.scatter(data.x[data.label == 2],data.y[data.label == 2 ],color = "blue")
