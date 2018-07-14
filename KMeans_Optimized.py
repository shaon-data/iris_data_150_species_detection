# -*- coding: utf-8 -*-
"""
Title: Getting Groups of species from unlabelled Iris Dataset
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing, cross_validation
from sklearn.cluster import KMeans

from EDA import *



## Constants
Resource = "data"
Result_Folder = "result"
FILE_NAME = Resource+"/iris_no_label.csv"

try:  
    os.mkdir(Result_Folder)
except OSError:  
    print ("Result Folder Already exist, overwriting everything.")

## Settings
style.use('ggplot')

sns.set()

    
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

    k_ticks = ["k"+str(k) for k in k_s]
    #ind = np.arange(len(range(2,15)))

    print(data.columns)

    def bins_labels(bins, **kwargs):
        bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
        print(bin_w)
        print(np.arange(min(bins)+bin_w/2, max(bins), bin_w))
        print(min(bins)+bin_w/2, max(bins), bin_w)
        print(np.arange(0.5,3, 1))
        plt.xticks(np.arange(0.5,3, 1), bins, **kwargs)
        #plt.xlim(bins[0], bins[-1])

    labels = data['k3_label'].unique()
    
    ## All possibilities with value of K
    fig = plt.figure("Scatter Matrix",figsize = (5,3))
    n=1
    for c in data.ix[:,4:].columns:
        ax = plt.subplot(5,3,n)
        labels, counts = np.unique(data[c], return_counts=True)
        ax.bar(labels, counts, align='center')
        #ax.gca().set_xticks(labels)
        ax.set_xlabel("Label")
        ax.set_ylabel("Label Population")
        
        n+=1
    plt.show()

    
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


    ## Plot of  Optimization starts
    plt.figure("k vs Model Cost and k vs Change rate in Model Cost")
    ## Plotting the k vs Model cost
    #plt.figure("k vs Model Cost(sum of distance from centroid)")
    plt.subplot(2,1,1)
    plt.plot(k_s,costs, marker = 'x')
    plt.title("Title:k vs Model Cost(sum of distance from centroid)")
    plt.xlabel('k')
    plt.ylabel('Model Cost')
    #plt.savefig(Result_Folder+"/k_vs_Model_Cost.png")

    ##d/dk(costs) = slope of Costs reference to K value = Rate of change of Costs reference to change of x
    M = slope_list_curve(k_s,costs)

    ## Visualizing optimized K value
    plt.subplot(2,1,2)
    #plt.figure("k vs d/dk(Cost)")
    plt.plot(k_s,M, marker = 'x')
    plt.title("Title:k vs Change_rate(Cost)")
    plt.xlabel('k')
    plt.ylabel('Change in Cost')
    
    plt.tight_layout()
    plt.savefig(Result_Folder+"/cost_ddk_costs.png")
    ## Plot of  Optimization ends
    

    ## Optimized Result
    best_k_index = M.index(min(M))
    best_k = k_s[best_k_index]
    best_cluster_number = nLabels[M.index(min(M))]
    
    
    

    ax = plt.figure('Best Result')
    labels, counts = np.unique(data['k'+str(k_s[best_k_index])+'_label'], return_counts=True)
    plt.title("Best Class Number=%s, when k=%s"%(best_cluster_number,best_k))
    plt.bar(labels, counts, align='center')
    plt.gca().set_xticks(labels)
    plt.xlabel("Label")
    plt.ylabel("Label Population")
    plt.savefig(Result_Folder+"/best_result.png")
    ax.show()
    

if __name__ == '__main__':
    main()
