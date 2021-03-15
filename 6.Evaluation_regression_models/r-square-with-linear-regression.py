# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import data
df = pd.read_csv("linear-regression-dataset.csv", sep = ";")
plt.scatter(df.deneyim,df.maas)
#plot data
plt.xlabel("deneyim")
plt.ylabel("maas")

#%% linear regression

# sklearn library
from sklearn.linear_model import LinearRegression

#linear regression model
linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

y_head = linear_reg.predict(x)
plt.plot(x, y_head , color = "red")


#%% 

from sklearn.metrics import r2_score

print("r_square score: ", r2_score(y, y_head))