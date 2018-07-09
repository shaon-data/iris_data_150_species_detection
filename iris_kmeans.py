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

FILE_NAME = "res/iris.csv"

## loading the data
data = pd.read_csv(FILE_NAME, header=None, index_col=0, names = ["sepal_width", "sepal_length", "petal_width", "petal_length", "Class"] )
data.reset_index(inplace=True)

data.fillna(0, inplace=True)

# Create an array of three colours, one for each species.
colors = np.array(['red', 'green', 'blue'])

#0,1,2 labels for selecting color by index
data.loc[ data['Class']=='Iris-setosa', 'Class'] = 0
data.loc[ data['Class']=='Iris-versicolor', 'Class'] = 1
data.loc[ data['Class']=='Iris-virginica', 'Class'] = 2

data = shuffle(data)

x = data.iloc[:,:-1]
y = data['Class']

#12x3 unit plot
plt.figure(figsize=(12,7))

#nrows=1, ncols=2, plot_number=1
plt.subplot(2, 2, 1)
plt.scatter(x['sepal_length'], x['sepal_width'], c=colors[y], s=40)
plt.title('Sepal Length vs Sepal Width')

#nrows=1, ncols=2, plot_number=2
plt.subplot(2,2,2)
plt.scatter(x['petal_length'], x['petal_width'], c= colors[y], s=40)
plt.title('Petal Length vs Petal Width')

clu = KMeans(n_clusters=3)
clu.fit(x)

print("Clustered plots %s" % clu.labels_)

#Start with a plot figure of size 12 units wide & 3 units tall


# Create an array of three colours, one for each species.
colors = np.array(['red', 'green', 'blue'])
labelss = np.array(['Setosa', 'Versicolor', 'Virginica'])

# The fudge to reorder the cluster ids.
#predictedY = np.choose(clu.labels_, [1, 0, 2]).astype(np.int64)
predictedY = clu.labels_
print("Clustered plots %s" % predictedY)

# Plot the classifications that we saw earlier between Petal Length and Petal Width
plt.subplot(2, 2, 3)
plt.scatter(x['petal_length'], x['petal_width'], label = labelss[y], c=colors[y], s=40) #colormap
plt.title('Actual Classes')
color_patch = []
for c in range(len(colors)):
    
    color_patch.append(mpatches.Patch(color=colors[c], label=labelss[c]))
    plt.legend(handles=color_patch)

# Plot the classifications according to the model
plt.subplot(2, 2, 4)
plt.scatter(x['petal_length'], x['petal_width'], c=colors[predictedY], s=40) #colormap
plt.title("Model's predicted classes")


plt.tight_layout()
plt.show()

'''
accuracy = clf.score(X_test, y_test)
print("Test Score %s" %accuracy)

predict = clf.predict(XP)
print(yP,np.round( np.array(predict)))
prediction_score = ( sum( yP==np.round( np.array(predict) ) ) / len(yP) ) * 100
print("Prediction score %s" %prediction_score)
'''
