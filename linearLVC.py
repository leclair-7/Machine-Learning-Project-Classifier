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
PreDataSetReduction = [residual[:-1] for residual in dataPre ]
data = sel.fit_transform(dataPre)  

print(len(sel.get_support() )  )
birch = sel.get_support() 
y = [ row[-1] for row in data]
X = [residual[:-1] for residual in data ]    
n_split = 1800    
X_train, X_test = X[:n_split], X[n_split:]
Y_train, Y_test = y[:n_split], y[n_split:]

clf = LinearSVC(penalty='l1', loss='squared_hinge', dual=False,tol=1e-3)
clf.fit(X_train, Y_train)

temp= clf.score(X_test, Y_test)
wer = clf.predict(X_test)
#Best LinearSVC Accuracy is:  [50, 0.9243055555555556]    
print("LinearSVC Accuracy is: ", temp)


