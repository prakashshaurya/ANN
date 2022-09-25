# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 13:52:42 2022

@author: shaurya NVIDIA EXAMPLE LEARNING DEEP LEARNING
"""

import numpy as np 

"""
x = vector whose firt element is always 1 
w = weight vector whose firat element is bias
length of x,w both be of (n+1)
"""

#numpy version
def perceptron_np(x,w) :
    z = np.dot(w.T,x) 
    if z<0 :
        return -1
    else :
        return 1


w= np.array([[0.9],
             [-0.6],
             [-0.5]],
            dtype='float')

x= np.array([[1.0],
             [-1.0],
             [-1.0]],
            dtype='float')

print(perceptron_np(x,w) )