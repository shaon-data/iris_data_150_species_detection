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
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

style.use('ggplot')


FILE_NAME = "res/iris.csv"

## loading the data
data = pd.read_csv(FILE_NAME, header=None, index_col=0, names = ["sepal_width", "sepal_length", "petal_width", "petal_length", "class"] )

## dealing with missing data
## replacing with outliers in not available values
data.fillna(-99999, inplace=True)

data.reset_index(inplace=True)

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

    data = shuffle(data)

    #gathering sample and training set
    percent = 10
    sample_size = int(data.shape[0]*(percent/100))
    sample = data.sample(sample_size)

    #data for prediction
    XP = np.array(sample.drop(['class'],1))
    XP = preprocessing.scale(XP)
    yP = np.array(sample['class'])

    #data for training and test
    data.drop(sample.index,inplace=True)
    X = np.array(data.drop(['class'],1))
    X = preprocessing.scale(X)
    y = np.array(data['class'])


    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    clf = LinearRegression()
    clf.fit(X_train, y_train) #train
            
    accuracy = clf.score(X_test, y_test) #test
    print("Test Score %s" %accuracy)

    predict = clf.predict(XP)
    print(yP,np.round( np.array(predict)))
    ps = ( sum( yP==np.round( np.array(predict) ) ) / len(yP) ) * 100
    print("Prediction score %s" %ps)
    accur.append(accuracy)
    pred.append(ps)

print(sum(accur)/len(accur), sum(pred)/len(pred))
