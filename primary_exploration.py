# -*- coding: utf-8 -*-
"""
Title: Iris Dataset exploration using Linear Regressio
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy
import scipy.stats


from matplotlib import style
#import matplotlib.patches as mpatch
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

style.use('ggplot')


FILE_NAME = "res/iris_complete.csv"

## loading the data
data = pd.read_csv(FILE_NAME, header=None, index_col=0, names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"] )
data.reset_index(inplace=True)

N,M = data.shape

'''
## Creating Missing data virtually
import random
for c in range(6):
    row_index = random.randint(0, N-1)
    row_cell = random.randint(0, M-1)
    print(data.iloc[row_index][row_cell])
    data.iloc[row_index][row_cell] = np.nan

    print(data.iloc[row_index][row_cell])
'''
labelss = np.array(['Undefined[Unbound down]','Setosa', 'Versicolor', 'Virginica','Undefined[Unbound up]'])
colors = np.array(['black','r','b','g','y'])

## dealing with missing data
## replacing with outliers in not available values
data.fillna(0, inplace=True)


## converting label to number
data.loc[ data['class']=='Iris-setosa', 'class'] = 1
data.loc[ data['class']=='Iris-versicolor', 'class'] = 2
data.loc[ data['class']=='Iris-virginica', 'class'] = 3


'''features = data.ix[:,:-1]
#standardizing data
X = StandardScaler().fit_transform(features)
label = data['class']

## ------- EDA --------
## Mean
print("Means of DataSet")
dt_mean =  features.apply(np.mean) ## removing last column class and applying np.mean
print(dt_mean)

## Medians
print("Medians of dataset")
dt_median =  features.apply(np.median) ## removing last column class and applying np.mean
print(dt_median)
'''

''' All distributions have normal distribution
def dist_fit_test(df,label):
    size = df.shape[0]
    x = scipy.arange(size)
    y = df#scipy.int_(scipy.round_(scipy.stats.vonmises.rvs(5,size=size)*47))
    plt.hist(y, bins=range(25), color='b', label=c)

    dist_names = ['gamma', 'beta', 'rayleigh', 'norm', 'pareto']

    for dist_name in dist_names:
        dist = getattr(scipy.stats, dist_name)
        param = dist.fit(y)
        pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) * size
        plt.plot(pdf_fitted, label=dist_name)
        plt.xlim(0,25)
    plt.legend(loc='upper right')
    plt.show()
for c in features:
    dist_fit_test(features[c],c)
'''



accur = []
pred = []
for c in range(20):
    ## loading the data
    data = pd.read_csv(FILE_NAME, header=None, index_col=0, names = ["sepal_width", "sepal_length", "petal_width", "petal_length", "class"] )

    ## dealing with missing data
    ## replacing with outliers in not available values
    ## data.fillna(-99999, inplace=True) ## We wanted to penalize bigger where missing data available but it destroys our model inacurately
    data.fillna(0, inplace=True)

    data.reset_index(inplace=True)

    ## converting label to number
    data.loc[ data['class']=='Iris-setosa', 'class'] = 1
    data.loc[ data['class']=='Iris-versicolor', 'class'] = 2
    data.loc[ data['class']=='Iris-virginica', 'class'] = 3

    
    data = shuffle(data)

    #gathering sample and training set
    percent = 10
    sample_size = int(data.shape[0]*(percent/100))
    sample = data.sample(sample_size)
    data.drop(sample.index,inplace=True)

    #data for prediction
    XP = np.array(sample.drop(['class'],1))
    XP = preprocessing.scale(XP)
    yP = np.array(sample['class'])

    #data for training and test
    X = np.array(data.drop(['class'],1))
    X = preprocessing.scale(X)
    y = np.array(data['class'])


    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    clf = LinearRegression()
    clf.fit(X_train, y_train)
            
    accuracy = clf.score(X_test, y_test)
    print("Test Score %s" %accuracy)

    predict = clf.predict(XP)
    print(yP,np.round( np.array(predict)))
    prediction_score = ( sum( yP==np.round( np.array(predict) ) ) / len(yP) ) * 100
    print("Prediction score %s" %prediction_score)

    
    accur.append(accuracy)
    pred.append(prediction_score)

print("Avarage Accuracy %s" % (sum(accur)/len(accur)))
print("Average Prediction Score %s"%(sum(pred)/len(pred)))

plt.figure('Supervised Classification by Integer label Forcasting - Linear Regression')
#Actual Points
plt.scatter(data['petal_length'], data['petal_width'], c=colors[y], s=40) #colormap

#Predicted Points
plt.scatter(sample['petal_length'], sample['petal_width'], c=colors[yP], s=200,marker='x') #colormap

color_patch = [Patch(color=colors[c], label=labelss[c]) for c in range(len(colors))]
color_patch.append(Line2D([0], [0], marker='o', color='w', label='Points', markerfacecolor='black', markersize=15))
color_patch.append(Line2D([0], [0], marker='X', color='w', label='Prediction', markerfacecolor='black', markersize=15))
plt.legend(handles=color_patch, numpoints=3)
plt.title("Determining Species of Plants from IRIS DATA")
plt.show()



