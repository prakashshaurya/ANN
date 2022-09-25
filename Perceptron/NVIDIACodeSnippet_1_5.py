# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 13:52:42 2022

@author: shaurya NVIDIA EXAMPLE LEARNING DEEP LEARNING
"""

import numpy as np 
import random
import matplotlib.pyplot as plt
"""
x = vector whose firt element is always 1 
w = weight vector whose firat element is bias
length of x,w both be of (n+1)
"""
color_list = ['r-','b-','g-','c-','m-','b-']
color_index =0


def perceptron(x,w) :
     z=0.0
     for i in range(len(x)) :
         z += x[i]*w[i]
     if z<0 :
          return -1
     else :
          return 1
w = [1.0 ,-1.0,-1.0]
x= [0.9,-0.6,-0.5]
print(perceptron(x,w))


def show_learning(w) :
    global color_index
    plt.plot([1.0,-1.0,-1.0] , [-1.0,-1.0,1.0] , 'r+' ,markersize=12)
    plt.plot([1.0] , [1.0] , 'b_' ,markersize=12)
    plt.axis([-2.0,2.0,-2.0,2.0])
    plt.xlabel('x1')
    plt.ylabel('x2')
    x=[-2.0,2.0]
    if abs(w[2]) <(1e-5) :
        y = [-w[1]*(-2)/ (1e-5) - w[0]/(1e-5) ,-w[1]*(2)/(1e-5) - w[0]/(1e-5) ] 
    else :
        y = [-w[1]*(-2)/w[2] - w[0]/w[2] ,-w[1]*(2)/w[2] - w[0]/w[2] ] 
    plt.plot(x,y,color_list[color_index-1])
    if color_index< (len(color_list)-1)    :
     color_index += 1
     #plt.show()
    print("w0 = %5.2f"% w[0],"w1 = %5.2f"% w[1],"w2 = %5.2f"% w[2])
    
    #define variables to control the training process 
    
random.seed(7)
LEARNING_RATE =0.1
index_list = [0,1,2,3]
x_train=[(1.0,-1.0,-1.0),(1.0,-1.0,1.0),(1.0,1.0,-1.0),(1.0,1.0,1.0)]
y_train=[1.0,1.0,1.0,-1.0]
w = [0.2,-0.6,0.25]

def training() :
    all_correct=False
    while not all_correct :
        all_correct =True
        random.shuffle(index_list)
        for i in index_list :
            x = x_train[i]
            y = y_train[i]
            p_out = perceptron(x,w)
            
            if y != p_out :
                for j in range(len(w)) :
                    w[j] = w[j] + (y*LEARNING_RATE*x[j])
                all_correct =False
                show_learning(w)
training() 
#show_learning(w)