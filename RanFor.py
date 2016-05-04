
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

'''
For variance threshold split 40
accuracy number 0.607777777778
number of classifiers 120
learning rate for classifiers 1
'''

maxAcc = [0,0.0,0, 0]
for r in range( 40,95,10):
    theIndex = r/ 100.0
    
    sel = VarianceThreshold(threshold=(theIndex* (1 - theIndex)))
    '''
    dataPre = np.loadtxt("train.nmv.txt")    
    data = sel.fit_transform(dataPre)
    
    y = [ row[-1] for row in data]
    X = [residual[:-1] for residual in data ]    
    n_split = 1800
    
    X_train, X_test = X[:n_split], X[n_split:]
    Y_train, Y_test = y[:n_split], y[n_split:]
    '''
    dataPre = np.loadtxt("train.nmv.txt")
    prelimData = np.genfromtxt("prelim-nmv-noclass.txt")
    prelimData = [i[:-1] for i in prelimData]
    
    prelimData = np.array(prelimData)
    
    classVec = [ row[-1] for row in dataPre]
    data = [residual[:-1] for residual in dataPre ]
    
    forFeaturetakeup = np.concatenate((data, prelimData), axis =0)    
    sel = VarianceThreshold(threshold=(theIndex * (1 - theIndex)))    
    data = sel.fit_transform(forFeaturetakeup)    
    prelimSet = data[4000:]
    data = data[:-4000]
    
    n_split = 1800
    X_test, X_train = data[:n_split], data[n_split:]
    Y_test, Y_train = classVec[:n_split], classVec[n_split:]

    p=0
    for numClassifiers in range(20,700,25):
        print(r, numClassifiers)
        p+=1
        model = AdaBoostClassifier(
            GaussianNB(),
            n_estimators=numClassifiers,
            learning_rate= 1 )

        model.fit(X_train, Y_train)

        tryBest = model.score(X_test,Y_test)
        if  tryBest > maxAcc[1]:
                maxAcc[0] = r                    
                maxAcc[1] = tryBest
                maxAcc[2] = numClassifiers
                maxAcc[3] = 1
print()
print("Best AdaBoosted GaussianNB Accuracy is: ")
print("Parameters are:")
print("For variance threshold split",maxAcc[0])
print("accuracy number", maxAcc[1])
print("number of classifiers",maxAcc[2])
print("learning rate for classifiers", maxAcc[3])

