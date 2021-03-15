# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:08:54 2020

@author: rhaen
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

#%%
iris = load_iris()

x = iris.data
y = iris.target

#%%
x = (x-np.min(x))/(np.max(x)-np.min(x))

#%% train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)

#%%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
# k  = 3 iyi bir değer çünkü accuracy %97
#%% K fold CSv K = 10
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=knn,X = x_train, y = y_train, cv = 10)
print("average accuracy: ",np.mean(accuracies))
print("average std: ",np.std(accuracies))

#%%
knn.fit(x_train,y_train)
print("test accuracy: ",knn.score(x_test,y_test))

#%% grid search cross validation

from sklearn.model_selection import GridSearchCV
grid = {"n_neighbors":np.arange(1,50)}
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv =10) #GridSearchCV
knn_cv.fit(x,y)

#%% print hyperparameter KNN algoritmasındaki K değeri

print("tuned hyperparameter K: ",knn_cv.best_params_)
print("tuned parametreye göre en iyi accuracy(best score): ",knn_cv.best_score_)

#%% Grid Search CV with Logistic Regression
x = x[:100,:]
y = y[:100]

from sklearn.linear_model import LogisticRegression

grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]} #Regularization parameter, C büyükse overfit, C çok küçükse underfit yani datayı modelleyememe öğrenememe olur
#l1 = Lasso ve l2 = ridge parameters

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, grid, cv =10)
logreg_cv.fit(x,y) #x_train ve y_train yapmalıydık

print("tuned hyperparameter (best parameters): ",logreg_cv.best_params_)
print("accuracy: ",logreg_cv.best_score_)