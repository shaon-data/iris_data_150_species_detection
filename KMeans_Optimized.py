# -*- coding: utf-8 -*-
"""
Title: Getting Groups of species from unlabelled Iris Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing, cross_validation
from sklearn.cluster import KMeans



## Constants
Resource = "res"
FILE_NAME = Resource+"/iris_no_label.csv"

## Settings
style.use('ggplot')

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

def max_min_bi_corelation(X):
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

def slope_list_curve( X, Y ):
        ## y = f(x)
        ## m = y2 - y1 / x2 - z1 = f(x2) - f(x1) / x2 - x1
        x1,y1 = 0,0
        M = []
        for x2,y2 in zip(X,Y):
            dy,dx = (y2 - y1),(x2 - x1)
            x1,y1 = x2,y2
            M.append( dy / dx )
        return M
    
def main():
    ## Loading the data
    data = pd.read_csv(FILE_NAME, header=None, index_col=0, names = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width'] )
    ## for unique ID for labels in optimization
    data.reset_index(inplace=True) # use this or remove, header = None from previous line
    
    ## Filling missing data with medians
    data = data.apply(lambda x: x.fillna(x.median()), axis=0 )
    

    ## Measures without no label
    X=data
    ## Reference dataset for comparing or reusing
    reference_data = data.copy()

    ## Getting Highest or least co-relation from covarience matrix
    #max_corelation_bi_measures, min_corelation_bi_measures = max_min_bi_corelation(X)

    ## Optimization: experimenting with differnt K values with their model costs
    k_s = []
    costs = []
    nLabels = []
    
    for k in range(2,15): ## experiment with n
        if True: ## Dont use Odd logic - if it is not continuous, we will not able to produce the real result
            ## Initializing model with a fixed random seed
            clusters = KMeans(n_clusters=k, random_state = 1)
            clusters.fit(X)
            ## Getting predicted Labels
            predictedLabelY = clusters.labels_

            ## Getting Model cost/inertia/sum of squared distance of data points from centroid
            cost = clusters.inertia_
            
            ## Genarating col name of K value for predicted labels
            col_name = 'k'+str(k)+'_label'
            ## Saving predicting labels
            
            data[col_name] = predictedLabelY
            ## Number of labels for specific K value

            ## Saving k value in every session
            k_s.append(k)
            ## Saving Number of labels for specific K value
            nLabels.append(data[col_name].nunique())
            ## Saving Cost or inertia for specific K value of clustering model
            costs.append(cost)

    ## shifting indexes to 1 row down
    data.index += 1

    ## Saving the labeled Result
    data.to_csv('res/unsupervised_label.csv')

    ## Plotting the k vs Number of labels to understand the cluster
    plt.figure("k vs Number of labels")
    plt.plot(k_s,nLabels, marker = 'x')
    plt.title("k vs label numbers")
    plt.xlabel('K')
    plt.ylabel('Number of labels')
    plt.savefig(Resource+"/k_vs_Number_of_labels.png")

    ## Plotting the k vs Model cost
    plt.figure("k vs Model Cost(sum of distance from centroid)")
    plt.plot(k_s,costs, marker = 'x')
    plt.title("k vs Model Cost(sum of distance from centroid)")
    plt.xlabel('k')
    plt.ylabel('Model Cost')
    plt.savefig(Resource+"/k_vs_Model_Cost.png")



    ##d/dk(costs) = slope of Costs reference to K value = Rate of change of Costs reference to change of x
    M = slope_list_curve(k_s,costs)

    ## Visualizing optimized K value    
    plt.figure("k vs d/dk(Cost)")
    plt.plot(k_s,M, marker = 'x')
    plt.title("k vs Change_rate(Cost)")
    plt.xlabel('k')
    plt.ylabel('Change in Cost')
    plt.savefig(Resource+"/ddk_costs.png")

    print(costs)
    plt.figure('HIst cost')
    plt.hist(costs, bins=100,alpha=0.75,normed=1)
    plt.xlabel('I')
    plt.ylabel('Costs')
    plt.title('HIst cost(Density) for Best Cluster Number[ later probability Dist]')
    plt.savefig(Resource+"/Histogram_costs.png")
    
    best_k_index = M.index(min(M))
    best_cluster_number = nLabels[M.index(min(M))]

    
    print("Best K = %s"%(k_s[best_k_index]))
    print("Best Label not ok = %s"%(best_cluster_number))
    print("Finished")

if __name__ == '__main__':
    main()
        

        
        
