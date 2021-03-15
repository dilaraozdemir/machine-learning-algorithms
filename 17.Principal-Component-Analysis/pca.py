""# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:51:45 2020

@author: rhaen
"""


from sklearn.datasets import load_iris
import pandas as pd

#%%

iris = load_iris()

data = iris.data
feature_names = iris.feature_names
y = iris.target #classlar

df = pd.DataFrame(data,columns = feature_names)
df["sinif"] = y

x = data

#%%
from sklearn.decomposition import PCA
pca = PCA(n_components = 2, whiten = True) # Whiten true diyince normalize ediliyor

pca.fit(x) # datanın featurelarıyla yapıyoruz. 4 boyutu 2ye düşür

x_pca = pca.transform(x)

# Hangisi second, hangisi principal
print("variance ratio: ",pca.explained_variance_ratio_)

print("sum:",sum(pca.explained_variance_ratio_))
# Sadece %3lük bir kayıp var
#%% 2D

df["p1"] = x_pca[:,0] #%92lik olan
df["p2"] = x_pca[:,1] #%5lik olan

color = ["red","green","blue"]

import matplotlib.pyplot as plt
for each in range(3):
    plt.scatter(df.p1[df.sinif == each],df.p2[df.sinif == each],color = color[each],label = iris.target_names[each])
plt.legend()
plt.xlabel("p1")
plt.ylabel("p2")
    