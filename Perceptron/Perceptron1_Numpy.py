# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 13:52:42 2022

@author: shaurya NVIDIA EXAMPLE LEARNING DEEP LEARNING
"""

import numpy as np 
import matplotlib.pyplot as plt
import random
import copy
"""
x = vector whose firt element is always 1 
w = weight vector whose firat element is bias
length of x,w both be of (n+1)
"""

#numpy version
def perceptron_np(x,w) :
    z = np.dot(w.T,x).astype(float) 
    if z<0 :
        return -1
    else :
        return 1


LEARNING_RATE =.1

index_list = [0,1,2,3]
x_train=np.array([[1.0,1.0,1.0],[1.0,-1.0,1.0],[1.0,1.0,-1.0],[1.0,-1.0,-1.0]])
"""
Here vector stored as row vector 
"""
#converting row matrix to column matrix
arr=np.array(x_train[0].reshape(-1,1))
  
y_train=np.array([[1.0],[-1.0],[-1.0],[-1.0]])
w_arr =np.array( [[0.2],[-0.6],[0.25]])
random.seed(7)


def learning(w_arr) :
    w_arr = copy.deepcopy(w_arr)
    
    for i in index_list :
        y =perceptron_np(x_train[i].reshape(-1,1),w_arr) 
        print("value of y is =",y )
        print("value of y_train is=", y_train[i,0])
        target = y_train[i][0]
    
        if y != target :
            print("before w0 = %5.2f"% w_arr[0],"w1 = %5.2f"% w_arr[1],"w2 = %5.2f"% w_arr[2])
            print ("learning occurs here")
            w_arr += LEARNING_RATE*y*(x_train[i].reshape(-1,1))
            print("after w0 = %5.2f"% w_arr[0],"w1 = %5.2f"% w_arr[1],"w2 = %5.2f"% w_arr[2])
            
    x=[-2,2]
    plt.plot([1.0,-1.0,-1.0] , [-1.0,-1.0,1.0] , 'r+' ,markersize=12)
    plt.plot([1.0] , [1.0] , 'b_' ,markersize=12)
    print(w_arr)
    y = [-w_arr[1]*(-2)/w_arr[2] - w_arr[0]/w_arr[2] ,-w_arr[1]*(2)/w_arr[2] - w_arr[0]/w_arr[2] ] 
    plt.plot(x,y)
    return w_arr
learning(w_arr)