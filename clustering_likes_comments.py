#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MeanShift, KMeans, estimate_bandwidth

le = preprocessing.LabelEncoder()

method = 'KMeans'

data = pd.read_csv('Live.csv', sep=',')

data['status_type'] = LabelEncoder().fit_transform(data['status_type'])
data['status_published'] = LabelEncoder().fit_transform(data['status_published'])

X = data.drop(['status_id', 'status_published', 'Column1', 'Column2', 'Column3', 'Column4'], axis=1)

if (method == 'MeanShift'):
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
    clustering = MeanShift(bandwidth=bandwidth).fit(X)
elif (method == 'KMeans'):
    clustering = KMeans(n_clusters=4, random_state=0).fit(X)
    
labels = clustering.labels_
cluster_centers = clustering.cluster_centers_

labels_unique = np.unique(labels)
n_clusters = len(labels_unique)
    
colors = ['b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k', 'b','g','r','c','m','y','k','b''g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k']

for i in range(n_clusters):
    A = X['num_comments']
    B = X['num_likes']
        
    plt.scatter(A[labels == i], B[labels == i], color=colors[i], zorder=0)
    
plt.xlabel('Number of comments')
plt.ylabel('Number of likes')
plt.show()
    
    
    
        
    
    
    
    

