from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn import tree
import Preprocessor
import pickle

class Classifier(BaseEstimator):
    def __init__(self):
        self.clf=None
        self.pipel = None
        pass

    ####### randomf #########
    def fit(self, X, y): 
        self.fit_randomForest(X, y)
        return
    
    def fit_ppl(self, X, y):
        clf = RandomForestClassifier()
        prep = Preprocessor()
        self.clf=Pipeline([('Prep', prep), ('classifieur', clf)])
        self.clf.fit(X, y)
	return
        
    ###############
    def predict(self, X):
        return self.clf.predict(X)
    
    ###############
    def predict_proba(self, X):
        return self.clf.predict_proba(X) # The classes are in the order of the labels returned by get_classes
    
    ###############
    def get_classes(self):
        return self.clf.classes_
    
    ###############
    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "w"))
    
    ###############
    def load(self, path="./"):
        self = pickle.load(open(path + '_model.pickle'))
        return self
    
    #################
    
    def fit_mlp(self, X, y):
        self.clf=MLPClassifier()
        for i in range(3):
            self.clf.fit(X, y)

    def fit_tree(self, X, y):
        self.clf = tree.DecisionTreeClassifier()
        self.clf.fit(X, y)
        
    def fit_nearNeighbors(self, X, y):
        self.clf = KNeighborsClassifier()
        self.clf.fit(X, y)

    def fit_gaussian(self, X, y):
        self.clf=GaussianProcessClassifier()
        self.clf.fit(X, y)
        
    def fit_randomForest(self, X, y):
        self.clf=RandomForestClassifier()
        self.clf.fit(X, y)