
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
r = 50.0    
theIndex = r/ 100.0

sel = VarianceThreshold(threshold=(theIndex* (1 - theIndex)))


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
       
#if may not need a deep copy
y = [ row[-1] for row in data]
X = [residual[:-1] for residual in data ]    
n_split = 1800    
X_train, X_test = X[:n_split], X[n_split:]
Y_train, Y_test = y[:n_split], y[n_split:]

'''
numClassifiers = 450
model = AdaBoostClassifier(
    GaussianNB(),
    n_estimators=numClassifiers,
    learning_rate= 1 )
'''
model.fit(X_train, Y_train)

temp = model.score(X_test,Y_test)
print("Accuracy on AdaBoosted GaussianNB is: ", temp)



