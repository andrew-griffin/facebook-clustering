#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from pandas.plotting import scatter_matrix

le = preprocessing.LabelEncoder()

data = pd.read_csv('Live.csv', sep=',', names=['id','type','date','reactions','comments','shares','likes','loves','wow','haha','sad','angry','Column1','Column2','Column3','Column4'], header=0)

data['type'] = LabelEncoder().fit_transform(data['type'])
data['date'] = LabelEncoder().fit_transform(data['date'])

X = data.drop(['id', 'date', 'Column1', 'Column2', 'Column3', 'Column4'], axis=1)

scatter_matrix(X)
plt.show()
    
    
    
        
    
    
    
    

