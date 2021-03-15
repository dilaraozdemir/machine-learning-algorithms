# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 17:02:30 2020

@author: rhaen
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("random-forest-regression-dataset.csv", sep = ";",header = None)

x = df.iloc[:,0].values.reshape(-1,1) #.values numpy'a çeviriyor
y = df.iloc[:,1].values.reshape(-1,1)

#%% 
from sklearn.ensemble import RandomForestRegressor

# n_ estimator = number of tree, random_state = Kodu çalıştırdığımızda her seferinde farklı bir sonuç alırız.
# Burada random state kullanınca yine random değerler vereceke am bir öncekini nasıl böldüysen öyle böl demek istiyoruz.

rf = RandomForestRegressor(n_estimators = 100,random_state = 42)

rf.fit(x,y)

print("How much price in 7.8 level: ",rf.predict([[7.8]]))

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = rf.predict(x_)

#visualize
plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color = "green")
plt.xlabel("tribun level")
plt.ylabel("ucret")