# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 14:11:49 2022

@author: shaurya
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

dataset = tf.keras.datasets.mnist
(xtrain,ytrain),(xtest,ytest) = dataset.load_data()
xtrain =xtrain/255.0
xtest=xtest/255.0

plt.figure()
for i in range (1,10):
    plt.subplot(330+i)
    plt.imshow(xtrain[i],cmap='gray')
    
model =tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=xtrain[0].shape))
model.add(tf.keras.layers.Dense(xtrain[0].shape[0]*xtrain[0].shape[1],activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='sigmoid'))

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
model.fit(xtrain,ytrain,epochs=10)
cost,acc=model.evaluate(xtest,ytest) 

#prediction 
y_prediction = model.predict(xtest)
digit = np.argmax(y_prediction[0])