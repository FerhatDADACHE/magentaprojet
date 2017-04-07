
#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
This is an example of program that preprocessed data.
It does nothing it just copies the input to the output.
Replace it with programs that:
    normalize data (for instance subtract the mean and divide by the standard deviation of each column)
    construct features (for instance add new columns with products of pairs of features)
    select features (see many methods in scikit-learn)
    re-combine features (PCA)
    remove outliers (examples far from the median or the mean; can only be done in training data)
"""

from sys import argv
from sklearn.base import BaseEstimator
#import numpy as np
from sklearn.preprocessing import StandardScaler #donne 
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Imputer
from zDataManager import DataManager # The class provided by binome 1
# Note: if zDataManager is not ready, use the mother class DataManager


class Preprocessor(BaseEstimator):
    def __init__(self):
        self.transformer = StandardScaler()
        self.norm = Normalizer()
        self.imp = Imputer()
        self.data=None

    def fit(self, X, y=None):
        self.data=X
        self.lbl=y
        
        self.data = self.fit_transform(X, y)
        
        return 

    def fit_transform(self, X, y):
        X_normalized = self.norm.fit_transform(X,y)
        X_imputed = self.imp.fit_transform(X_normalized,y)
        return self.transformer.fit_transform(X_imputed, y)

    def transform(self, X, y=None):
        X_normalized = self.norm.transform(X)
        X_imputed = self.imp.transform(X_normalized)
        return self.transformer.transform(X_imputed)
    
if __name__=="__main__":
    # We can use this to run this file as a script and test the Preprocessor
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = "../public_data"
        output_dir = "../res"
    else:
        input_dir = argv[1]
        output_dir = argv[2];
    
    basename = 'crime'
    D = DataManager(basename, input_dir) # Load data
    print("*** Original data ***")
    print D
    
    Prepro = Preprocessor()
 
    # Preprocess on the data and load it back into D
    D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
    #D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
    D.data['X_valid'] = Prepro.transform(D.data['X_valid'])
    D.data['X_test'] = Prepro.transform(D.data['X_test'])
    #D.data['X_train'] = Prepro.fit_transform(D.data['X_train'], D.data['Y_train'])
    #D.data['X_valid'] = Prepro.get_dummies(D.data['X_valid'])
    #D.data['X_test'] = Prepro.get_dummies(D.data['X_test'])
  
    # Here show something that proves that the preprocessing worked fine
    print("*** Transformed data ***")
    print D
    
