
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


maxAcc = [0,0.0]
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
    
    clf = LinearSVC(penalty='l1', loss='squared_hinge', dual=False,tol=1e-3)
    clf.fit(X_train, Y_train)
    
    if r % 10 == 0:
        print(r)
    temp= clf.score(X_test, Y_test)
    if temp > maxAcc[1]:
        maxAcc[1] = temp
        maxAcc[0] = r
#Best LinearSVC Accuracy is:  [50, 0.9243055555555556]    
print("Best LinearSVC Accuracy is: ", maxAcc)


