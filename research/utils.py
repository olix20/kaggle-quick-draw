import os
from glob import glob
import re
import ast
import numpy as np 
import pandas as pd
from PIL import Image, ImageDraw 
from tqdm import tqdm_notebook,tqdm
from dask import bag
import dask 

import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras.layers import * 
from keras.models import *
from keras.callbacks import *
from keras.optimizers import Adam


from sklearn.model_selection import StratifiedShuffleSplit
from matplotlib import pyplot as plt


from keras.utils import Sequence
from imgaug import augmenters as iaa
import imgaug as ia

import math
import pickle
import gc


def preds2catids(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])

def save_obj(obj, path ):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path ):
    with open(path, 'rb') as f:
        return pickle.load(f)

def top_3_accuracy(x,y): 
	t3 = top_k_categorical_accuracy(x,y, 3)
	return t3


def get_model_with_imagenet_weights(imgnet_model):
	base_model = imgnet_model
	# add a global spatial average pooling layer
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	# let's add a fully-connected layer
	x = Dense(512, activation='relu')(x)
	x = Dropout(0.5)(x)
	predictions = Dense(num_classes, activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	for layer in base_model.layers:
	    layer.trainable = False	

	return model



def rotate_point(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy