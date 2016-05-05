# -*- coding: utf-8 -*-
"""
Created on Wed May  4 03:26:55 2016

@author: Lucas
"""

'''
By: Lucas Hagel
and Geoffrey Greenleaf
'''


from sklearn.svm import LinearSVC
'''
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.utils import check_random_state
from sklearn import datasets
'''
from sklearn.feature_selection import VarianceThreshold

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

'''
make a random 20/80 for train/test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
'''

#did it update automatically?
r = 49
theIndex = r/ 100.0
sel = VarianceThreshold(threshold=(theIndex * (1 - theIndex)))

data = np.loadtxt("train.nmv.txt")
firstData = data.copy()
tagUno = [ row[-1] for row in data]
tagUno = np.array([tagUno])
#arr = np.concatenate(  (arr , for_arr.T ), axis =1)

'''
Idea : -Save class labels which will be used
after the fit_transform thing 
cuts the poor variance labels down
'''    
data = sel.fit_transform(data)
data = np.concatenate(  (data , tagUno.T ), axis =1)
guillotine = sel.get_support()

prelimData = np.genfromtxt("prelim-nmv-noclass.txt")
prelimData = [i[:-1] for i in prelimData]    
prelimData = np.array(prelimData)

#guillotine = guillotine[:-1]
#assert len(guillotine) == len(prelimData[:-1])
guillotine_full = guillotine.copy()
guillotine = guillotine[:-1]

'''
#assigning this to prelimData verifies that we collapse the
#preliminary test set correctly
prelimData = firstData.copy()[:-1]
'''

for i in range(len(guillotine)):
    #print(i, guillotine[i])
    if guillotine[len(guillotine)-1-i] == False:        
       prelimData = np.delete(prelimData, len(guillotine)-1-i, axis=1)
       
#print(len(firstData[0]),len(data[0]),len(guillotine), len(prelimData[0]))

#if may not need a deep copy
y = [ row[-1] for row in data]
X = [residual[:-1] for residual in data ]    
n_split = 1800    
X_train, X_test = X[:n_split], X[n_split:]
Y_train, Y_test = y[:n_split], y[n_split:]

clf = LinearSVC(penalty='l1', loss='squared_hinge', dual=False,tol=1e-3)
clf.fit(X_train, Y_train)

temp= clf.score(X_test, Y_test)
predictionOfPrelim = clf.predict(prelimData)

#Best LinearSVC Accuracy is:  [50, 0.9243055555555556]    
print("LinearSVC Accuracy is: ", temp)

prelimClasses = np.loadtxt("prelim-class.txt")
assert len(prelimClasses) == len(predictionOfPrelim)
h = []
for i in range(len(prelimClasses)):
    if prelimClasses[i] == predictionOfPrelim[i]:
        h.append(1)
    else:
        h.append(0)

thefile = open('Result_LinearSVC_prelim_.txt', 'w')
for item in h:
  thefile.write("%s\n" % item)
  
print("ding")
'''
import matplotlib.pyplot as plt
plt.plot(h)
plt.ylabel('one is right, zero is wrong')
plt.show()
'''

