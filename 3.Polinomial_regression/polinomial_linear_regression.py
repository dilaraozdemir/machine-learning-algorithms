# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:17:26 2020

@author: rhaen
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("polynomial-regression.csv",sep = ";")

y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("araba_max_hiz")
plt.ylabel("aba_fiyat")


#linear regression = y = b0+b1*x
#multiple regression = y = b0 + b1*x1 + b2*x2

#%% Linear regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)
#%% predict
y_head = lr.predict(x)

plt.plot(x,y_head,color="red",label = "linear")
print("10 bin tllik araba hizi tahmini: ",lr.predict([[10000]]))
#%% polynomial regression = y = b0 + b1*x + b2*x^2 + b3*x^3 + ... + bn*x^n

from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = PolynomialFeatures(degree = 2)

x_polynomial = polynomial_regression.fit_transform(x)
#%% fit
linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial,y)  

#%% visualize

y_head2 = linear_regression2.predict(x_polynomial)

plt.plot(x,y_head2,color = "green", label = "poly")
plt.legend()