# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 11:35:41 2022

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
index = 25
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

# m_train = training portion of data
# m_test =  testing portion of data
# num_px =  shape of first image  = (64, 64, 3)
m_train = train_set_x_orig.shape[0]
m_test  = test_set_x_orig.shape[0]
num_px  = train_set_x_orig[1].shape[0]

# train_set_x_flatten = flattening RGB to Linear 
# test_set_x_flatten  = flattening RGB to Linear
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

#standardizing data
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

def sigmoid(z) :
    return ( 1/(1+np.exp(-z)))
#print(sigmoid(.5))

def dimSet(m) :
    w=np.zeros((m,1),dtype='float')
    b=0.0
    return (w,b)
#print(dimSet(20))

def propagate(w,b,X,Y) :
    m=X.shape[1]
    A= sigmoid(np.dot(w.T,X)+b)
    cost = np.sum((-1/m)*(Y*np.log(A) + (1-Y)*np.log(1-A)),axis=1)
    dw = (1/m)*np.sum(np.matmul(X,(A-Y).T))
    db = (1/m)*np.sum((A-Y),axis=1)
    grads ={"dw":dw,"db":db}
    return grads,cost
    
def learning(w,b,X,Y,learning_rate=0.009,num_itr=100,print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    grads,cost =  propagate(w,b,X,Y)
    costs=[]
    for i in range(num_itr) :
        dw = grads["dw"]
        db = grads["db"]
        w = w-learning_rate*dw
        b = b-learning_rate*db
        if i%100 ==0 :
            costs.append(cost)
            if print_cost:
                 print ("Cost after iteration %i: %f" %(i, cost))
    params ={"w":w,"b":b}
    grads = {"dw":dw,"db":db}
    return params ,grads,costs

def predict(w,b,X):
    m = X.shape[1]
    Y_predictions = np.zeros((1,m) )
    w = w.reshape(X.shape[0],1)
    Z = np.dot(w.T,X)+b
    A =sigmoid(Z)
    for i in range(A.shape[1]):
        if  A[0,i]>0.5 :
            Y_predictions[0,i] =1
        else:
            Y_predictions[0,i] =0
    return Y_predictions

def model (X_train,X_test,Y_train,Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False) :
    w,b = dimSet(X_train.shape[0])
    params ,grads,costs =learning(w,b,X_train,Y_train,learning_rate=0.009,num_itr=100,print_cost=False)
    w = params["w"]
    b = params["b"]
    Y_predictions_train= predict(w,b,X_train)
    Y_predictions_test = predict(w,b,X_test)
    # Print train/test Errors
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_predictions_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_predictions_test - Y_test)) * 100))
    
    d = {"costs": costs,
         "Y_prediction_test": Y_predictions_test, 
         "Y_prediction_train" : Y_predictions_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
print(model (train_set_x,test_set_x,train_set_y,test_set_y, num_iterations=2000, learning_rate=0.5, print_cost=False) )
costs = np.squeeze(model (train_set_x,test_set_x,train_set_y,test_set_y, num_iterations=2000, learning_rate=0.5, print_cost=False)['costs'])