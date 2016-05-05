# -*- coding: utf-8 -*-
"""
Created on Thu May  5 16:05:07 2016

@author: Lucas
"""

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


import numpy as np

data = np.loadtxt("train.nmv.txt")

prelimData = np.genfromtxt("prelim-nmv-noclass.txt")
prelimData = [i[:-1] for i in prelimData]    
prelimData = np.array(prelimData)

prelimClasses = np.loadtxt("prelim-class.txt")
prelimClasses = np.array([prelimClasses])

fullPrelim = np.concatenate( (prelimData, prelimClasses.T), axis = 1)
AllData = np.concatenate((data, fullPrelim),axis=0)

y = [ row[-1] for row in AllData ]
X = [ residual[:-1] for residual in AllData ]    
n_split = 2000

X_train, X_test = X[:n_split], X[n_split:]
Y_train, Y_test = y[:n_split], y[n_split:]

    
clf = ExtraTreesClassifier()

clf = clf.fit(X_train, Y_train)

numFeatures = 150
rfe = RFE(clf, numFeatures)
rfe = rfe.fit(X_train,Y_train)
temp = rfe.score(X_test, Y_test)
predictionOfPrelim = rfe.predict(prelimData)
featureRanking = rfe.ranking_
someOtherParam = clf.oob_score_

newthing = clf.feature_importances_
print('ExtraTrees Training accuracy:', clf.score(X_train, Y_train))
print('ExtraTrees Test accuracy:', clf.score(X_test, Y_test))

