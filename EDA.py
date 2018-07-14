# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class ValueKeeper(object):
    def __init__(self, value): self.value = value
    def __str__(self): return str(self.value)

class A(ValueKeeper):
    def __pos__(self):
        ## print('called A.__pos__')
        self.value += 1
        return self.value
    def __neg__(self):
        ## print('called A.__pos__')
        self.value -= 1
        return self.value
    def __mul__(self,a):
        return self.value*a

    
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

def squared_error(y,y_estimated):
    y = np.array(y)
    y_estimated = np.array(y_estimated)
    return sum( (y - y_estimated)**2 )
    ## squared_error

def least_square(y,y_estimated):
    y = np.array(y)
    y_estimated = np.array(y_estimated)
    d = y - y_estimated
    ## [0, infinity )
    return sum(d*d)

def r_squared(y,y_estimated):
    ## equivalent to least_square(y,y_estimated)
    SEy_estimated = squared_error(y,y_estimated) 
    SEy_mean = squared_error(y,meann(y)) 
    return 1 - (SEy_estimated/SEy_mean)
    ## coefficient of determination  or R^2, better than least square

'''
Another Less convinient methoood
def r_squared2(y,y_estimated,c=0):
    ## y_estimated or y hat
    y = np.array(y)
    y_estimated = np.array(y_estimated)
    y_mean = meann(y)
    Distance_between_estimated_line_N_Mean = y_estimated - y_mean ## = > This one is
    Distance_between_actual_point_N_Mean = y - y_mean # same
    ## both distance are squared , because sum of (actual line - mean) = 0 and we need distance so we have to avoid '-' numbers
    if c == 1:
        print("R = 0 no colinearity and R = 1 is exact feet, R = [0,1]\n we can say R sqaured , describes how well regression line, predicts actual values")
    rsquared = sum( Distance_between_estimated_line_N_Mean**2 ) / sum( Distance_between_actual_point_N_Mean**2 )
    ## between few lines y_estimated - y_mean
    return rsquared
'''
    
def standard_error(y,y_estimate):
    n = len(y)
    y = np.array(y)
    y_estimated = np.array(y_estimated)
    return ( sum( (y_estimated - y)**2 ) / (n - 2) )**(1/2)


def variance(x):
    ## S squared
    ## Measure of -  How spread the data is - solve the problem of outlier
    ## How spread and far your data points to each other, variance will get bigger
    ## by escaping or reducing the sensivitiy of outlier problem (for squaring which act like absolute value)
    x_mean = meann(x)
    xi = np.array(x)
    n = len(x) ## sample size
    ## deviation_score_of_x = xi - x_mean
    ## Sum of deviation scores is always = 0, so we square each individual deviation score and sum them later , like least square
    ## least square is , sum of each square of residual sum(residuali^2) where i = [1,n] n for population, n-1 for sample
    ## deviation = distance from mean
    ## SSD = Sample Standard deviation = average distance from mean
    S_squared = sum( (xi - x_mean) ** 2 ) / (n - 1)
    return S_squared
def pearson_r(x,y):
    x = np.array(x)
    y = np.array(y)
    Mx = meann(x)
    My = meann(y)
    pearson_r = sum( (x-Mx) * (y - My) ) / ( sum( (x - Mx)**2 ) * sum( (y - My )**2 ) )**(1/2)
    return pearson_r

def standard_deviation(x):
    ## S - Average distance from the mean
    return (variance(x))**(1/2)

def covarience_matrix(X):
    #standardizing data
    X_std = StandardScaler().fit_transform(X)

    #sample means of feature columns' of dataset
    mean_vec = np.mean(X_std, axis=0) 
    #covariance matrix
    ##[ (distance of data points from their mean)^T . (distance of data points from their mean) ] / ( n - 1 )
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

def scatter_matrix_graph_fit(data,s=8):
    
    measurement_number = data.shape[1]
    n=0
    fig = plt.figure("Scatter Matrix",figsize = (measurement_number,measurement_number))
    plt.axes(frameon=False)
    
    xS = 1
    for yI in data.columns:
        j=0
        yS = 1
        for xI in data.columns:
            yS = 1
            n+=1
            ax = plt.subplot(measurement_number,measurement_number,n)
            ax.scatter(data[xI],data[yI],c='mediumseagreen',s=s)
            y_hat = regression_points(data[xI],data[yI])
            ax.plot(data[xI],y_hat,c='deepskyblue',ls='-.')
            ax.set_title("r="+str(pearson_r(data[xI],data[yI])),fontsize=10,y=.89)

         

            #ax.axes.get_xaxis().set_visible(False)
            #ax.axes.get_yaxis().set_visible(False)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
            if j == 0:            
                ax.set_ylabel(yI)
                ax.set_xlabel(xI)
            else:
                ax.set_xlabel(xI)
            
            j+=1
            yS+=1
        xS+=1
    
    plt.subplots_adjust(wspace=0.02, hspace=0.08)
    plt.show()
    
    # plt.subplot(441, facecolor='y')

    
def standard_deviation_residuals(y,y_estimated):
    ## Standard deviation of residuals or Root mean sqaure error
    ## Lower the number is, the better the fit of the model
    ## Average distance to the mean
    n = len(y)
    return ( least_square(y,y_estimated) / (n - 1) )**(1/2)
    

def bivariate_regression_line_coefficients(x,y):
    x_mean = meann(x)
    y_mean = meann(y)
    n = len(x)
    x, y = np.array(x), np.array(y)
    b1 = ( sum( (x - x_mean) * (y - y_mean) ) ) / (sum( (x - x_mean)**2 ))
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


def good_fit_equation_lr_test():
    ## Determining best equation for linear regression
    ## outliers in data make range uselsess
    x = [43,21,25,42,57,59,247]
    y = [99,65,79,75,87,81,486]

    x_mean = meann(x)
    y_mean = meann(y)
    n = len(x)
    x, y = np.array(x), np.array(y)


    b1eq_ = ["b1 = ( sum( (x - x_mean) * (y - y_mean) ) ) / (sum( (x - x_mean)**2 ))","b1 = ( n*(sum(x*y)) - sum(x)*sum(y) ) / ( n*(sum(x**2)) - (sum(x))**2 )","b1 = ( sum( y*x ) - ( ( sum( y ) * sum( x ) ) / n ) ) / ( sum(  (x - x_mean)**2  ) )"]
    b0eq_ = ["b0 = ( sum(y)*sum(x**2) - sum(x)*sum(x*y) ) / ( n*sum(x**2) - (sum(x))**2 )","b0 = y_mean - b1*x_mean"]


    b0_,b1_ = [],[]
    b1 = ( sum( (x - x_mean) * (y - y_mean) ) ) / (sum( (x - x_mean)**2 )) #least0
    b1_.append(b1)
    b1 = ( n*(sum(x*y)) - sum(x)*sum(y) ) / ( n*(sum(x**2)) - (sum(x))**2 )
    b1_.append(b1)
    b1 = ( sum( y*x ) - ( ( sum( y ) * sum( x ) ) / n ) ) / ( sum(  (x - x_mean)**2  ) )
    b1_.append(b1)    

    print("Eq for b1=%s"%b1eq_)
    print("b1=%s"%b1_)

    b0 = ( sum(y)*sum(x**2) - sum(x)*sum(x*y) ) / ( n*sum(x**2) - (sum(x))**2 )
    b0_.append(b0)
    b0 = y_mean - b1*x_mean #least0
    b0_.append(b0)
    print("Eq for b0=%s"%b0eq_)
    print("b0 = %s"%b0_)

    y_estimated_ = []




    str_ = []
    c=1
    for b1i in b1_:
        for b0i in b0_:
            y_estimated_.append( b0i + b1i*x )
            ## print("#%s eq, y = %s + %s*x"%(c, b0i , b1i))
            str_.append("#%s eq, y = %s + %s*x"%(c, b0i , b1i))
            c+=1

    y_eqs_ = []
    for b1eq in b1eq_:
        for b0eq in b0eq_:
            y_eqs_.append( (b0eq, b1eq) )

    ## r_squared(y,y_estimated)
    '''
    c=1
    for y_estimated in y_estimated_:    
        print("#%s eq least square = %s"%(c,least_square(y,y_estimated)))
        c+=1
    '''
    '''
    c=A(0)
    [print("#%s eq least square = %s"%(+c,least_square(y,y_estimated))) for y_estimated in y_estimated_]
    '''

    [print(c,l,'R Squared = ',r,s,'\n Equations = ',eq) for l,r,c,s,eq in sorted(zip( [least_square(y,y_estimated) for y_estimated in y_estimated_],[r_squared(y,y_estimated) for y_estimated in y_estimated_],['#'+str(c)+' Least Square=' for c in range(1,6+1)],str_,y_eqs_))]

    import matplotlib.pyplot as plt
    plt.scatter(x,y)
    plt.plot(x,(3.6895964091445137 + 1.9153296055384377*x ))

        

    plt.show()

'''
x = [43,21,25,42,57,59,247]
y = [99,65,79,75,87,81,486]
newDF = pd.DataFrame()
newDF['x'] = x
newDF['y'] = y

print(pearson_r(x,y))
print(np.corrcoef(x,y))
print(covarience_matrix(newDF))
'''
