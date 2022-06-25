# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:10:47 2022

@author: shaurya
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 12:06:50 2022

@author: shaurya
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import pandas as pd

# Loading the data (cat/non-cat)
def load_dataset():
    with h5py.File('./train_catvnoncat.h5', "r") as train_dataset:
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    return train_set_x_orig
train_set = load_dataset()

# train set has 209 images , 64 X 64 Each , its a dataset.
# train_set.shape[0] = 209
# train_set.shape[1] = 64
# train_set.shape[2] = 64
# train_set.reshape(train_set.shape[0],-1).T 
# train_set.reshape(209,-1).T
flatten_image = train_set.reshape(train_set.shape[0],-1).T

#flatten_image is dataset of 209 images vectors each having 12288 i,e 12288X209 matrix

after_flattening=  pd.DataFrame(flatten_image)