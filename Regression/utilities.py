import numpy as np
import pandas as pd

def compute_cost(x,y,w,b):
    
    m = x.shape[0]  
    cost = 0.0 
    for i in range(m):
        prediction = np.dot(x[i],w) + b 
        error_distance = (y[i] - prediction)**2
        cost = cost + error_distance
    cost = cost/(2*m) # we devide by 2m rather then m its just for easy implementation of gradient descent 
    return cost 

def compute_gradient(x,y,w,b):
    """
    x = ndarray (m,n) m no of row , n of column
    y = ndarray (m,)
    w = ndarray (n) 1D vector with m element
    b is scalar 
    err is scalar
    """
    m,n= x.shape
    dj_dw = np.zeros(n,)
    dj_db = 0.0

    for i in range(m):
        err =  (np.dot(x[i],w) + b) - y[i]          
        dj_db = dj_db + err

    dj_db = dj_db/m
    dj_dw = dj_dw/m
    return dj_dw,dj_db 

def compute_gradient_matrix(x,y,w,b):
    """
    x = ndarray (m,n) m no of row , n of column
    y = ndarray (m,)
    w = ndarray (n) 1D vector with m element
    b is scalar

    Returns
    dj_dw (ndarray (n,1)): The gradient of the cost w.r.t. the parameters w.
    dj_db (scalar):        The gradient of the cost w.r.t. the parameter b.

    """
    
    m,n = x.shape
    prediction = x @ w + b
    err =  prediction - y
    dj_dw =  (1/m)*(x.T@err)#.T is transpose of x we did it cause matrix multiplication take place between dimension (m,n)(n,m)
    # here error is of dimension (m,) 
    dj_db = (1/m)*np.sum(err)

    return dj_dw,dj_db

def compute_cost_matrix(x,y,w,b):
    """
    Computes the gradient for linear regression
    Args:
    X (ndarray (m,n)): Data, m examples with n features
    y (ndarray (m,)) : target values
    w (ndarray (n,)) : model parameters  
    b (scalar)       : model parameter
    verbose : (Boolean) If true, print out intermediate value f_wb
    Returns
    cost: (scalar)
    """
    x = x.to_numpy() if isinstance(x, (pd.DataFrame, pd.Series)) else x
    y = y.to_numpy() if isinstance(y, (pd.DataFrame, pd.Series)) else y
    w = np.array(w, dtype=float)
    m=x.shape[0]
    prediction = x@w + b
    err  =    (y - prediction)**2
    cost =  (1/(2*m))*(np.sum(err))

    return cost

def fit(x,y,w,b,alfa,epocs):
    x = x.to_numpy() if isinstance(x, (pd.DataFrame, pd.Series)) else x
    y = y.to_numpy() if isinstance(y, (pd.DataFrame, pd.Series)) else y
    w = np.array(w, dtype=float)
    cost_history = []
    m = x.shape[0]
   
    for i in range(epocs):
        pred = x @ w + b
        error = pred - y

        dj_dw = (1/m)*((x.T)@error)
        dj_db = (1/m)*np.sum(error)

        cost = compute_cost_matrix(x,y,w,b)
        cost_history.append(cost)

        w = w - alfa*dj_dw
        b = b - alfa*dj_db
        
       

    return w,b,cost_history

def testTrainSplit(x,y,train_size,seed=None):
    x = x.to_numpy().astype(float)if isinstance(x, (pd.DataFrame, pd.Series)) else x
    y = y.to_numpy().astype(float)if isinstance(y, (pd.DataFrame, pd.Series)) else y
 
    if seed :
        np.random.seed(seed)
    m = x.shape[0]
    indices = np.random.permutation(m)

    count = int(m * train_size)

    train_data= indices[:count] 
    test_data= indices[count:] 

    return x[train_data],x[test_data],y[train_data],y[test_data]

def scaling_data(xTrain,xTest):
    
    meanX=[]
    stdX =[]
    for i in range(xTrain.shape[1]):
        meanFeature =  np.mean(xTrain[:,i])
        meanX.append(meanFeature)
        stdFeature = np.std(xTrain[:,i])
        stdX.append(stdFeature)
   
    xTrain = helper(xTrain,meanX,stdX)
    xTest = helper(xTest,meanX,stdX)
    return xTrain,xTest,meanX,stdX

def helper(data,mean,std):
    n = data.shape[1]
    for i in range(n):
        meanF = mean[i]
        starndard_deviation = std[i]    
        if starndard_deviation == 0:
            starndard_deviation = 1  
        data[:,i] = (data[:,i]-meanF)/starndard_deviation
       
    return data
   
def predict(x,w,b):
    x = x.to_numpy() if isinstance(x, (pd.DataFrame, pd.Series)) else x
    w = np.array(w, dtype=float)
    return x@w + b

