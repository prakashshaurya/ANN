# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 11:46:57 2022

@author: shaurya
"""

import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

train_set_x_flatten = train_set_x/255
test_set_x_flatten = test_set_x/255

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def weight(dim):
    w = np.zeros((dim,1),dtype='float')
    b = 0.0
    return w,b

def propagate(X,w,b,Y) : 
    m = train_set_x_flatten.shape[1]  # m=12288
    w,b = weight(m)
    A = sigmoid(np.dot(w.T,X)+b)  # A = [a1,a2,a3......a209]
    cost = (1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A),keepdims=True)
    dw =(1/m)* np.matmul(X,(A-Y).T)
    db =(1/m)* np.sum((A-Y),axis=1)
    cost = np.squeeze(np.array(cost))
    grads = {"dw":dw,"db":db}
    return grads,cost

def optimize(X_train,Y_train,LEARNING_RATE=0.01,num_iterations=100,print_cost=False) :
    m =X_train.shape[1]
    w,b = weight(m)
    costs =[]
    for i in range(num_iterations) :
        grads,cost = propagate(X_train,w,b,Y_train)
        dw = grads["dw"]
        db = grads["db"]
        w[i] = w[i] -LEARNING_RATE*dw*X_train[i]
        b[i] = b[i] -LEARNING_RATE*db
        if i%100 == 0 :
            costs.append(cost)
        if print_cost :
            print(costs)
        params = {"w":w ,"b":b}
        grads = {"dw":dw,"db":db}
    return 
 
def predict (w,b,X):
    m = X.shape[1] # no of examples
    Y_prediction = np.zeros(1,m)
    w= w.rehsape(X.shape[1],1)
    A = sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]) :
        if (A[0,i]) >0.5:
            Y_prediction[0,i] = 1
        else :
            Y_prediction[0,i] = 0
        return Y_prediction

def model (X_train,Y_train,X_test,Y_test,LEARNING_RATE=0.5,num_iterations=2000,print_cost=False) :
    pass