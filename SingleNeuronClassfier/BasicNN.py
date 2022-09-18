# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 12:32:07 2022

@author: shaurya
"""
import numpy as np

def sigmoid(x) :
    s = 1/(1+np.exp(-x))
    return s

def derivativeSig(x) :
    s= np.exp(-x)/(1+np.exp(-x))**2
    return s

"""
axis =1 means row wise 
axis =0 means column wise , ord =1,2,3....  
x=x/norm means broadcasting of a value
"""
def normalize_rows(x) :  
    norm = np.linalg.norm(x,axis=1,keepdims=True)
    x=x/norm
    return x
x = np.array([[0, 3, 4],
              [1, 6, 4]])
print("normalizeRows(x) = " + str(normalize_rows(x)))


"""
evaluate softmax , keepdoims=True avoids reshaping
"""
def softmax(x) :
   
    x_exp = np.exp(x)
    norm = np.linalg.norm(x_exp,axis=1,keepdims=True)
    x_exp=x_exp/norm
    return x_exp
t_x = np.array([[9, 2, 5, 0, 0],
                [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(t_x)))


yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])

def L1(yhat,y) :
   lossVal = np.sum(np.absolute(yhat-y)) 
   return lossVal
   
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat, y)))





