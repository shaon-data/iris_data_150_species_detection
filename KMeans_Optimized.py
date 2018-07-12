# -*- coding: utf-8 -*-
"""
Title: Getting Groups of species from unlabelled Iris Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style


## Constants
FILE_NAME = "res/iris_no_label.csv"

## Settings
style('ggplot')

## Functions
def covarience_matrix(X):
    #standardizing data
    X_std = StandardScaler().fit_transform(X)
    #sample means of feature columns' of dataset
    mean_vec = np.mean(X_std, axis=0)
    #covariance matrix
    ##[ (distance of data points from their mean)^T . (distance of data points from their mean) ] / ( n - 1 )
    cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
    ## Equivalent code from numpy: cov_mat = np.cov(X_std.T)
    
    ## if size of dataset = n x m = number of samples[row] x Measurements[column]
    ## m = number of mesurements
    ## m x m will be the number of cc-relation elemnt returned as 2D matrix, as it is 2 or bivariate
    ## max number is more corelated or less number is less corelated
    return cov_mat

def max_min_bi_corel(X):
    ## Max and Min bivariate co-relation from covarience matrix
    a = covarience_matrix(X)
    ''' Converting diagonal of covariance matrix from 1 to 0.
    cov(measureX,measureX) => Variance of  Element vs Element = 1
    which is distributed amoung diagonal.
    That means diagonals denotes fully corelated situation.
    So we don't need the diagonal, converting them to 0 '''
    a[a>=1] = 0
    #Max corelation
    maxcor = np.argwhere(a.max() == a)[0] # reverse 1
    b = covarience_matrix(X)
    #Min corelation
    mincor = np.argwhere(b.min() == b)[0] # reverse 1
    return maxcor,mincor



