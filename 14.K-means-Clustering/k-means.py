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
x1 = np.random.normal(25,5,1000) # 25 ortalamaya ve 5 sigmaya sahip 1000 tane dğer üret
y1 = np.random.normal(25,5,1000)

# Class2
x2 = np.random.normal(55,5,1000) # 25 ortalamaya ve 5 sigmaya sahip 1000 tane dğer üret
y2 = np.random.normal(60,5,1000)

# Class3
x3 = np.random.normal(55,5,1000) # 25 ortalamaya ve 5 sigmaya sahip 1000 tane dğer üret
y3 = np.random.normal(15,5,1000)

x = np.concatenate((x1, x2, x3),axis=0)
y = np.concatenate((y1, y2, y3),axis=0)

dictionary = {"x":x,"y":y}

data = pd.DataFrame(dictionary)

#%%
plt.scatter(x1,y1,color = "black")
plt.scatter(x2,y2,color = "black")
plt.scatter(x3,y3,color = "black")

#%% K-MEANS

from sklearn.cluster import KMeans
wcss = []

for k in range(1,15):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,15),wcss)
plt.xlabel("number o k (cluster) value")
plt.ylabel("wcss")
plt.show()

#%% K = 3 için model

kmeans2 = KMeans(n_clusters =3)
Clusters = kmeans2.fit_predict(data) # fit ettiğin değeri datanın üzerinde uygula ve clusterları oluştur.

data["label"] = Clusters

plt.scatter(data.x[data.label == 0],data.y[data.label == 0 ],color = "red")
plt.scatter(data.x[data.label == 1],data.y[data.label == 1 ],color = "green")
plt.scatter(data.x[data.label == 2],data.y[data.label == 2 ],color = "blue")
plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color = "yellow")

