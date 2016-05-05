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

'''
r = 400
theIndex = r/ 1000.0
sel = VarianceThreshold(threshold=(theIndex* (1 - theIndex)))
'''

data = np.loadtxt("train.nmv.txt")
firstData = data.copy()
tagUno = [ row[-1] for row in data]
tagUno = np.array([tagUno])
#arr = np.concatenate(  (arr , for_arr.T ), axis =1)

data = sel.fit_transform(data)
data = np.concatenate(  (data , tagUno.T ), axis =1)
guillotine = sel.get_support()

prelimData = np.genfromtxt("prelim-nmv-noclass.txt")
prelimData = [i[:-1] for i in prelimData]    
prelimData = np.array(prelimData)


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
predictionOfPrelim = rfe.predict(prelimData)

featureRanking = rfe.ranking_
#Best ExtraTrees Accuracy is:  [400, 0.98902777777777773, 40]            
print("ExtraTrees Accuracy is: ", temp)

prelimClasses = np.loadtxt("prelim-class.txt")
assert len(prelimClasses) == len(predictionOfPrelim)
h = []
for i in range(len(prelimClasses)):
    if prelimClasses[i] == predictionOfPrelim[i]:
        h.append(1)
    else:
        h.append(0)

thefile = open('Result_ExtraTrees_prelim.txt', 'w')
for item in h:
  thefile.write("%s\n" % item) 



# display the relative importance of each attribute
#print(model.feature_importances_)
'''
print('Training accuracy:', model.score(X_train, Y_train))
print('Test accuracy:', model.score(X_test, Y_test))
'''