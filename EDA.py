# -*- coding: utf-8 -*-
import numpy as np
    
def meann(y):
    return sum(y)/len(y)

def error(y,population_mean):
    y = np.array(y)
    population_mean = np,array(population_mean)
    return y - population_mean

def residual(y,y_estimated):
    ## y_estimated can be y regression line(estimated point) or single y mean
    ## y_estimated or regression points are not given, it take out y mean from y
    y = np.array(y)
    if y == empty():
        y_estimated = np,array(meann(y))
    else:
        y_estimated = np.array(y_estimated)
    return y - y_estimated

def least_square(y,y_estimated):
    y = np.array(y)
    y_estimated = np.array(y_estimated)
    d = y - y_estimated
    ## [0, infinity )
    return sum(d*d)

def r_squared(y,y_estimated,c=0):
    ## y_estimated or y hat
    y = np.array(y)
    y_estimated = np.array(y_estimated)
    y_mean = meann(y)
    Distance_between_estimated_line_N_Mean = y_estimated - y_mean
    Distance_between_actual_line_N_Mean = y - y_mean
    ## both distance are squared , because sum of (actual line - mean) = 0 and we need distance so we have to avoid '-' numbers
    if c == 1:
        print("R = 0 no colinearity and R = 1 is exact feet, R = [0,1]\n we can say R sqaured , describes how well regression line, predicts actual values")
    rsquared = sum( (Distance_between_estimated_line_N_Mean)**2 ) / sum( ( y - y_mean )**2 )
    return rsquared
    
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
    '''
    ## [x,y,z,...]  * [x,y,z,...] = [[ xx xy xz .....] = [[ 1  xy xz .....]
                                     [ yx yy yz .....]    [ yx 1  yz .....]
                                     [ zx zy zz .....]    [ zx zy 1 .....]
                                     [...............]    [...............]
                                     ...............      ...............
                                    ]                     ]
    1 perfect fit or colinearity , 0 no colinearity  [0,1]
    '''                                
    return cov_mat

def max_min_bi_corel(X):
    a = covarience_matrix(X)
    a[a>=1] = 0
    maxcor = np.argwhere(a.max() == a)[0] # reverse 1

    b = covarience_matrix(x)
    mincor = np.argwhere(b.min() == b)[0] # reverse 1

    return maxcor,mincor

def standard_error(y,y_estimate):
    n = len(y)
    y = np.array(y)
    y_estimated = np.array(y_estimated)
    return ( sum( (y_estimated - y)**2 ) / (n - 2) )**(1/2)

def standard_deviation_residuals(y,y_estimated):
    ## Standard deviation of residuals or Root mean sqaure error
    ## Lower the number is, the better the fit of the model
    n = len(y)
    return ( least_square(y,y_estimated) / (n - 1) )**(1/2)
    

def bivariate_regression_line_coefficients(x,y):
    x_mean = meann(x)
    y_mean = meann(y)
    x, y = np.array(x), np.array(y)
    b1 = ( sum( (x - x_mean) * (y - y_mean) ) ) / (sum( (x - x_mean)**2 ))
    '''
    n = len(y)
    b1 = ( sum( y*x ) - ( ( sum( y ) * sum( x ) ) / n ) ) / ( sum(  (x - x_mean)**2  ) )
    '''    
    b0 = y_mean - b1*x_mean
    ## y_estimated = b0 + b1*x
    return b0,b1
    

def regression_points(x,y):
    ## best fit line point
    b0,b1 = bivariate_regression_line_coefficients(x,y)
    x = np.array(x)
    y_estimated = b0 + b1*x
    return y_estimated

def handling_missing_data(data):
    ## data = dataframe
    #data.fillna(0, inplace=True)
    #data = data.apply(lambda x: x.fillna(x.mean()),axis=0)
    data = data.apply(lambda x: x.fillna(x.median()),axis=0)

    ## if one parameter is missing checck the another parameter with colinearity.
    ## Check the second parameter probability distribution, you will have the missing data should be 0/mean/median or max
    ## Not sure but practice the procedure.

x=[1,2,2,3]
y=[1,2,3,6]
y_estimated = regression_points(x,y)

print(least_square(y,y_estimated))





