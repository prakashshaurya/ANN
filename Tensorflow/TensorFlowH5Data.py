# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 12:44:10 2022

@author: shaurya
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from lr_utils import load_dataset
import h5py
# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
x_train = train_set_x_orig/255.0
x_test =test_set_x_orig/255.0
model = tf.keras.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(64,64,3)))
model.add(tf.keras.layers.Dense(10, activation='sigmoid'))
model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
model.fit(x_train ,train_set_y.T ,epochs=35)
cost,acc = model.evaluate(test_set_x_orig,test_set_y.T,verbose=2)
