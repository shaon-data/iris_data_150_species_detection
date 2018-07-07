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



## taking the first 4 sample by default
## print(data.head())
## taking the 4 random sample
## print(data.sample(4))



## taking the missing data
print("Showing the missing data")
print(data.isna())
## if everything is right procced

