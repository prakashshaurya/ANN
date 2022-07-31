# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 20:30:05 2022

@author: shaurya
"""
import numpy as np

class LogisticRegression :
    def __init__(self,w,b) :
        self.w=w
        self.b=b
    def __str__(self) :
        return f"zvalue = {self.z} and bvalue ={self.b}"
    def getW(self) :
        return self.w
    def getB(self) :
        return self.b
    
def evalZ(logisticReg,x):
    return np.dot(logisticReg.getW(),x)+logisticReg.getB()

def sigmoid(z) :
    a = 1/(1+np.exp(-z))
    return a
    
w = np.random.rand(3,4) 

logistic = LogisticRegression(1,1)
z=(evalZ(logistic,.3))
print(sigmoid(z))