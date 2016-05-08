from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.naive_bayes import GaussianNB
import numpy as np

from collections import namedtuple

def remove_variance(features, p):
    t = (p * (1 - p))
    sel = VarianceThreshold(threshold=p)
    sel = sel.fit(features)
    return sel.fit_transform(features), sel

def train_data_and_score_tree(features,labels, cv, depth):
    f_train, f_test, l_train, l_test = cross_validation.train_test_split(
        features, labels, test_size=cv,random_state=0
    ) 

    clf = ExtraTreesClassifier(max_depth=depth)
    # clf = DecisionTreeClassifier(max_depth=depth)
    clf = clf.fit(f_train,l_train)
    score = clf.score(f_test,l_test)
    
    return score,clf

def train_data_and_score_bayes(features, labels, cv):
    f_train, f_test, l_train, l_test = cross_validation.train_test_split(
        features, labels, test_size=cv,random_state=0
    ) 
    gnb = GaussianNB()
    score = gnb.fit(f_train,l_train).score(f_test,l_test)
    return score
    
def select_n_features(features, labels, n):
    sel = SelectKBest(f_classif,k=n)
    new_features = sel.fit_transform(features, labels)
    return new_features
    
results = []
best_results = []
ResTree = namedtuple("ResultTree",["cv","depth","variance","score","clf","sel"],verbose=True)
# ResBayes = namedtuple("ResultBayes",["cv","variance","score"],verbose=True)

data = np.loadtxt("AllData.txt") 
features, labels = [x[:-1] for x in data], [x[-1] for x in data]

for i in xrange(10):
    print i
    for var in np.arange(.7,1,.5):
        for cv in np.arange(.2,.5,.05):
            for depth in range(20,100,5) + [None]:
                new_features,sel = remove_variance(features, var)       
                score, clf = train_data_and_score_tree(new_features,labels, cv, depth)
                results.append(ResTree(cv,depth, var, score,clf,sel))
    best_clf = max(results, key=lambda x: x.score)
    best_results.append(best_clf)        
                    

best_clf = max(best_results, key=lambda x: x.score)
print best_clf

#Getting predictions for test set with non missing values
final_test_set = np.loadtxt("final-nmv-noclass.txt",dtype=str)
final_features = [x[:-1] for x in final_test_set]
final_features = np.array(final_features)

sel = best_clf.sel
clf = best_clf.clf

new_final_features = sel.transform(final_features)
predictions = clf.predict(new_final_features)
with open("final-predictions.txt","w") as f:
    f.write("\n".join(str(int(x)) for x in predictions))
    


   
    

