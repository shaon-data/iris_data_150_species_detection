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

from pandas.tools.plotting import scatter_matrix
from mpl_toolkits.mplot3d import Axes3D
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
    ## for unique id for labels
    data.reset_index(inplace=True)
    

    # Create an array of three colours, one for each species.
    colors = np.array(['red', 'green', 'blue','yellow','violet','orange','black','pink','purple','crimson','brown'])
    kmarker = np.array(["v","o","*","."])
    labelss = np.array(['0', '1', '2','3','4','5','6','7','8'])

    
    #def handle_missing_values()
    #data.fillna(0, inplace=True)
    #data = data.apply(lambda x: x.fillna(x.mean()),axis=0)
    data = data.apply(lambda x: x.fillna(x.median()),axis=0)

    #data = shuffle(data)

    x = data
    reference_data = data.copy()

    print("Covariance Matrix =")
    print(covarience_matrix(x))

    maxcor,mincor = max_min_bi_corel(x)
    print("Most Colinearity %s"%maxcor)
    print("Least Colinearity %s"%mincor)

    
    k_=[]
    t_=[]
    t_c=[]
    kl_n = []
    k_inertia = []
    for k  in range(2,15):
        if k%2 != 0:
            clu = KMeans(n_clusters=k,random_state =1)
            clu.fit(x)
            predictedY = clu.labels_
            print("k=%s , center = "%k)

            # Sum of distances of samples to their closest cluster center
            interia = clu.inertia_
            k_inertia.append((interia)**(1/2))
            
            print(clu.cluster_centers_)
            bata = data[['sepal_width','petal_length','petal_width']]
            scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='kde')
            # Plot the classifications according to the model
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            
            
            ax.scatter(x['sepal_width'][predictedY == 0],x['petal_length'][predictedY == 0], x['petal_width'][predictedY == 0],  c=x['sepal_length'][predictedY == 0],cmap=plt.hot(), label='A', marker=(5, 0, 45)) #colormap
            ax.scatter(x['sepal_width'][predictedY == 1],x['petal_length'][predictedY == 1], x['petal_width'][predictedY == 1],  c=x['sepal_length'][predictedY == 1],cmap=plt.hot(), label='B', marker=(5, 2, 45)) #colormap
            ax.scatter(x['sepal_width'][predictedY == 2],x['petal_length'][predictedY == 2], x['petal_width'][predictedY == 2],  c=x['sepal_length'][predictedY == 2],cmap=plt.hot(), label='C', marker=(5, 1, 45)) #colormap
            ax.scatter(0,0,label='Color Hitmap',c='w')
            
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
            ax.legend()

            '''
            ax.scatter(x['sepal_width'][predictedY == 0],x['petal_length'][predictedY == 0], x['petal_width'][predictedY == 0],  c='y', marker=(5, 0, 45)) #colormap
            ax.scatter(x['sepal_width'][predictedY == 1],x['petal_length'][predictedY == 1], x['petal_width'][predictedY == 1],  c='g',cmap=plt.hot(), marker=(5, 2, 45)) #colormap
            ax.scatter(x['sepal_width'][predictedY == 2],x['petal_length'][predictedY == 2], x['petal_width'][predictedY == 2],  c='r',cmap=plt.hot(), marker=(5, 1, 45)) #colormap
            '''

            ## taking n-dimensional centroid and plotting centroid
            c0=[]
            c3=[]
            c2 = []

            for c in clu.cluster_centers_:
                ax.scatter(c[0],c[3], c[2], c='blue', marker = 'D', s=100) #colormap
                c0.append(c[0])
                c3.append(c[3])
                c2.append(c[2])
            #ax.plot(sorted(c0),sorted(c3),sorted(c2))
            plt.title("Model's predicted classes[Clusters without label]")

            plt.tight_layout()
            plt.show()
            
        
            # The fudge to reorder the cluster ids.
            # predictedY = np.choose(clu.labels_, [1, 0, 2]).astype(np.int64)
            predictedY = clu.labels_
            col_name = 'k'+str(k)+'_label'
            data[col_name] = predictedY
            k_.append(k)
            kl_n.append(data[col_name].nunique())
        
        '''
        target_labels = np.unique(clu.labels_)
        
        ula = []
        for t in target_labels:
            #print(k,t,predictedY.tolist().count(t))
            ula.append(predictedY.tolist().count(t))
            
        k_.append(n)
        
        
        #sorting algorithm for sorting least distance from previous class        
        
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

        #sorting algorithm for sorting least distance from previous class
        '''
        
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
        


    print(k_,t_,t_c)
    sales = {'k': k_,
         'Counts': t_c,
            }
    
        
    my_df = pd.DataFrame.from_dict(sales)
    '''
    ## start indexing from 1
    data.index += 1 
    data.to_csv('res/unsupervised_label.csv')
    #[for c in data.columns[:4]]

    data_labels = data.copy()
    for c in data_labels.columns[:4]:
        data_labels.drop([c],axis=1,inplace=True)
    data_labels.to_csv('res/only_unsupervised_label.csv')

    ## finding the k vs labels count
    for c in data_labels.columns:
        print(data_labels[c].nunique())

        
    ### print(" Sort %s" %sorted(t_c))

    plt.figure("k vs label")
    plt.plot(k_,kl_n, marker = 'x')
    plt.xlabel('k')
    plt.ylabel('Number of labels')
    plt.title("k vs label numbers")

    plt.figure("k vs sum of distance from centroid")
    plt.plot(k_,k_inertia, marker = 'x')
    plt.xlabel('k')
    plt.ylabel('sum of distance from centroid')
    plt.title("k vs sum of distance from centroid")
    plt.show()

    ## determine and plot the k vs elbow point
    
    #for c in 
    
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
    
