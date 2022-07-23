# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:16:07 2022

@author: shaurya
"""
import numpy as np
import h5py

arr = np.random.rand(4,4)

### element wise application ###

expo = np.exp(arr)

divide = 1/arr

absolute = np.abs(arr)

maximu = np.maximum(arr[0],0)

multiply= arr**2

#train set
Dataset= h5py.File('./train_catvnoncat.h5', "r") 
train_x=np.array(Dataset["train_set_x"][:])
flatten_train_x = train_x.reshape(train_x.shape[0],-1).T
#test set
Dataset_test = h5py.File('./test_catvnoncat.h5', "r") 
test_x =np.array(Dataset_test["test_set_x"][:])
flatten_test_x=(test_x.reshape(test_x.shape[0],-1)).T

def sigmoid(x):
    return (1/(1+np.exp(-x)))

print(sigmoid(0))

def initWB(dim) :
    w = np.zeros((dim,1),dtype='f')
    b=0.0
    return (w,b)
print(initWB(10))

def evaluator(w,b,X,Y) :
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)  #outcome A= [y1,y2,y3,y4..............]
    cost = -(1/m )*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A), axis=1,keepdims=True)
    dw = (1/m)*np.dot(X,(A-Y).T)
    db = (1/m)*np.sum((A-Y))
    grads = {"dw":dw, "db":db}
    return grads,cost

w,b=initWB(12288)
print(w,b)

Y=np.ones((1,209))

grads,cost=evaluator(w,b,flatten_train_x,Y)

dw= grads["dw"]
flatten_train_x = flatten_train_x -.5*dw*flatten_train_x