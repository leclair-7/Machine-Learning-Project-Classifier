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


r = 49
theIndex = r/ 100.0
sel = VarianceThreshold(threshold=(theIndex * (1 - theIndex)))    

dataPre = np.loadtxt("train.nmv.txt")    
data = sel.fit_transform(dataPre)  

guillotine = sel.get_support()

print("data len: ", len(data[0]))

prelimData = np.genfromtxt("prelim-nmv-noclass.txt")
prelimData = [i[:-1] for i in prelimData]    
prelimData = np.array(prelimData)

assert len(guillotine) == len(prelimData[0])

for i in range(len(guillotine)):
    #print(i, guillotine[i])
    if guillotine[len(guillotine)-1-i] == False:        
       prelimData = np.delete(prelimData, len(guillotine)-1-i, axis=1) 
        
print("data len: ", len(prelimData[0]))


'''
arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(arr)
arr = np.delete(arr, 1, axis=1)
print("postDeletion")
print(arr)

'''

