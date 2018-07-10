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
import matplotlib.patches as mpatches
from sklearn import preprocessing, cross_validation, svm
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.cluster import KMeans

style.use('ggplot')

FILE_NAME = "res/iris_no_label.csv"

def covarience_matrix(X):
    #standardizing data
    X_std = StandardScaler().fit_transform(X)

    #sample means of feature columns' of dataset
    mean_vec = np.mean(X_std, axis=0) 
    #covariance matrix
    cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
    #if right handside is ( Xstd - mean(Xstd) )^T . ( Xstd - mean(Xstd) )
    #simplyfying X^T.X / ( n - 1 )
    cov_mat = np.cov(X_std.T)
    return cov_mat

def max_min_bi_corel(X):
    a = covarience_matrix(X)
    a[a>=1] = 0
    maxcor = np.argwhere(a.max() == a)[0] # reverse 1

    b = covarience_matrix(X)
    mincor = np.argwhere(b.min() == b)[0] # reverse 1

    return maxcor,mincor

def main():
    ## loading the data
    data = pd.read_csv(FILE_NAME, header=None, index_col=0, names = ["sepal_width", "sepal_length", "petal_width", "petal_length"] )
    data.reset_index(inplace=True)
    

    # Create an array of three colours, one for each species.
    colors = np.array(['red', 'green', 'blue','yellow','violet','orange','black','pink'])
    labelss = np.array(['0', '1', '2','3','4','5','6','7','8'])

    
    #def handle_missing_values()
    #data.fillna(0, inplace=True)
    #data = data.apply(lambda x: x.fillna(x.mean()),axis=0)
    data = data.apply(lambda x: x.fillna(x.median()),axis=0)

    #data = shuffle(data)

    x = data

    print("Covariance Matrix =")
    print(covarience_matrix(x))

    maxcor,mincor = max_min_bi_corel(x)
    print("Most Colinearity %s"%maxcor)
    print("Least Colinearity %s"%mincor)


    lb = pd.read_csv('res/iris_complete.csv', header=None, index_col=0, names = ["sepal_width", "sepal_length", "petal_width", "petal_length", "Class"] )
    yyy = lb['Class'].tolist()
    
    n_=[]
    t_=[]
    t_c=[]
    for n in range(2,9):
        clu = KMeans(n_clusters=n,random_state =1)
        clu.fit(x)

        

        # The fudge to reorder the cluster ids.
        #predictedY = np.choose(clu.labels_, [1, 0, 2]).astype(np.int64)
        predictedY = clu.labels_
        target_labels = np.unique(clu.labels_)
        ula = []
        for t in target_labels:
            #print(n,t,predictedY.tolist().count(t))
            ula.append(predictedY.tolist().count(t))
            
        n_.append(n)
        
        
        
        
        if n != 2:
            sula = []
            for xd in cula:
                distance_frod_xd = np.absolute(xd - np.array(ula))
                
                
                dis_ula = sorted(zip( distance_frod_xd, ula ))
                
                _,fu = dis_ula[0]
                
                sula.append(fu)
            dula = sula+list(set(ula)-set(sula))
            print(list(set(ula)-set(sula)))
            
            cula=dula
        else:
            cula=ula
        
        t_c.append(cula)

        '''
        #print("Clustered groups %s" % predictedY)
        #12x7 unit plot
        plt.figure('Unsupervised Clustering - KMeans',figsize=(12,7))
        
        #nrows=1, ncols=2, plot_number=1
        plt.subplot(2, 2, 1)
        plt.scatter(x[data.columns[mincor[0]]], x[data.columns[mincor[1]]], c=colors[predictedY], s=40)
        plt.title('%s vs %s[Lowest Colinear]'%(data.columns[mincor[0]],data.columns[mincor[1]]))

        #nrows=1, ncols=2, plot_number=2
        plt.subplot(2, 2, 2)
        plt.scatter(x[data.columns[maxcor[0]]], x[data.columns[maxcor[1]]], c=colors[predictedY], s=40)
        plt.title('%s vs %s[Highest Colinear]'%(data.columns[maxcor[0]],data.columns[maxcor[1]]))


        color_patch = []
        for c in range(len(colors)):    
            color_patch.append(mpatches.Patch(color=colors[c], label=labelss[c]))
            plt.legend(handles=color_patch)

        # Plot the classifications according to the model
        plt.subplot(2, 2, 4)
        plt.scatter(x['petal_length'], x['petal_width'], c=colors[predictedY], s=40) #colormap
        plt.title("Model's predicted classes[Clusters without label]")

        plt.tight_layout()
        plt.show()
        '''


    print(n_,t_,t_c)
    sales = {'n': n_,
         'Counts': t_c,
            }
    my_df = pd.DataFrame.from_dict(sales)
    my_df.to_csv('res/n.csv')
    print(" Sort %s" %sorted(t_c))

    
    '''
    accuracy = clf.score(X_test, y_test)
    print("Test Score %s" %accuracy)

    predict = clf.predict(XP)
    print(yP,np.round( np.array(predict)))
    prediction_score = ( sum( yP==np.round( np.array(predict) ) ) / len(yP) ) * 100
    print("Prediction score %s" %prediction_score)
    '''

if __name__ == '__main__':
    main()
    
