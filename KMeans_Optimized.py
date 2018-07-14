# -*- coding: utf-8 -*-
"""
Title: Getting Groups of species from unlabelled Iris Dataset
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing, cross_validation
from sklearn.cluster import KMeans

from EDA import *



## Constants
Resource = "res"
Result_Folder = "result"
FILE_NAME = Resource+"/iris_no_label.csv"

try:  
    os.mkdir(Result_Folder)
except OSError:  
    print ("Result Folder Already exist, overwriting everything.")

## Settings
style.use('ggplot')

    
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

    scatter_matrix_graph_fit(data)
    
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
    data.to_csv(Result_Folder+'/unsupervised_label.csv')

    ## Plotting the k vs Number of labels to understand the cluster
    plt.figure("k vs Number of labels")
    plt.plot(k_s,nLabels, marker = 'x')
    plt.title("k vs label numbers")
    plt.xlabel('K')
    plt.ylabel('Number of labels')
    plt.savefig(Result_Folder+"/k_vs_Number_of_labels.png")

    ## Plotting the k vs Model cost
    plt.figure("k vs Model Cost(sum of distance from centroid)")
    plt.plot(k_s,costs, marker = 'x')
    plt.title("k vs Model Cost(sum of distance from centroid)")
    plt.xlabel('k')
    plt.ylabel('Model Cost')
    plt.savefig(Result_Folder+"/k_vs_Model_Cost.png")



    ##d/dk(costs) = slope of Costs reference to K value = Rate of change of Costs reference to change of x
    M = slope_list_curve(k_s,costs)

    ## Visualizing optimized K value    
    plt.figure("k vs d/dk(Cost)")
    plt.plot(k_s,M, marker = 'x')
    plt.title("k vs Change_rate(Cost)")
    plt.xlabel('k')
    plt.ylabel('Change in Cost')
    plt.savefig(Result_Folder+"/ddk_costs.png")

    print(costs)
    plt.figure('HIst cost')
    plt.hist(costs, bins=100,alpha=0.75,normed=1)
    plt.xlabel('I')
    plt.ylabel('Costs')
    plt.title('HIst cost(Density) for Best Cluster Number[ later probability Dist]')
    plt.savefig(Result_Folder+"/Histogram_costs.png")
    
    best_k_index = M.index(min(M))
    best_cluster_number = nLabels[M.index(min(M))]

    
    print("Best K = %s"%(k_s[best_k_index]))
    print("Best Label not ok = %s"%(best_cluster_number))
    print("Finished")

if __name__ == '__main__':
    main()