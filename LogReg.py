# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:19:23 2016

@author: Lucas
"""


# Recursive Feature Elimination
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# load the iris datasets
from sklearn.feature_selection import VarianceThreshold

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


maxAcc = [0,0.0,0]


for r in range( 40,90,10):
    theIndex = r/ 100.0
    sel = VarianceThreshold(threshold=(theIndex * (1 - theIndex)))
    
    
    dataPre = np.loadtxt("train.nmv.txt")    
    data = sel.fit_transform(dataPre)
    
    y = [ row[-1] for row in data]
    X = [residual[:-1] for residual in data ]    
    n_split = 1800
    
    X_train, X_test = X[:n_split], X[n_split:]
    Y_train, Y_test = y[:n_split], y[n_split:]
    

    for numFeatures in range(40,200,40):
        if numFeatures % 10 ==0:
            print("testing  ", numFeatures,"features,",r ," is variance tested")
        model = LogisticRegression()
        # create the RFE model and select 3 attributes
        rfe = RFE(model, numFeatures)
        rfe = rfe.fit(X_train,Y_train)
        
        temp = rfe.score(X_test, Y_test)
        
        if temp > maxAcc[1]:
            maxAcc[0] = r
            maxAcc[1] = temp
            maxAcc[2] = numFeatures
# [84, 0.70777777777777773, 130]
            
#Best Logistic Regression Accuracy is:  [50, 0.93013888888888885, 80]
print("Best Logistic Regression Accuracy is: ", maxAcc)
