# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:15:57 2016

@author: Lucas

variance threshold with an ExtraTreesClassifier
-- brute force search of good parameters -- 
"""

from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

r = 400

theIndex = r/ 1000.0

sel = VarianceThreshold(threshold=(theIndex* (1 - theIndex)))


dataPre = np.loadtxt("train.nmv.txt")    
data = sel.fit_transform(dataPre)

y = [ row[-1] for row in data]
X = [residual[:-1] for residual in data ]    
n_split = 1800

X_train, X_test = X[:n_split], X[n_split:]
Y_train, Y_test = y[:n_split], y[n_split:]

numFeatures = 40  
    
model = ExtraTreesClassifier()

#model.fit(X_train, Y_train)
rfe = RFE(model, numFeatures)
rfe = rfe.fit(X_train,Y_train)

temp = rfe.score(X_test, Y_test)

if temp > maxAcc[1]:
    maxAcc[0] = r
    maxAcc[1] = temp
    maxAcc[2] = numFeatures
#Best ExtraTrees Accuracy is:  [400, 0.98902777777777773, 40]            
print("Best ExtraTrees Accuracy is: ", maxAcc)



# display the relative importance of each attribute
#print(model.feature_importances_)
'''
print('Training accuracy:', model.score(X_train, Y_train))
print('Test accuracy:', model.score(X_test, Y_test))
'''