# -*- coding: utf-8 -*-
"""
Title: Iris Dataset exploration using Linear Regression


"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE_NAME = "res/iris.csv"

## loading the data
data = pd.read_csv(FILE_NAME, header=None, index_col=0, names = ["sepal_width", "sepal_length", "petal_width", "petal_length", "class"] )

## dealing with missing data
## replacing with outliers in not available values
data.fillna(-99999, inplace=True)

## Taking the missing data
data.reset_index(inplace=True)
print("Showing the missing data")
'''
    isnull converting to null T/F matrix, then if any element is null,
    it is true, then pick it to our new data matrix and print them.
'''
print(data[data.isnull().any(axis=1)==True])
## resetting back the index as first column
data.set_index('sepal_width',inplace=True)
## if everything is right procced

print(data.head())
