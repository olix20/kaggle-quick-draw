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

from keras.metrics import top_k_categorical_accuracy
from keras.utils import multi_gpu_model

from ast import literal_eval


import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.sequence import pad_sequences

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




import numpy as np
import gc
import keras

from keras.applications.mobilenet_v2 import MobileNetV2
from keras.optimizers import Adam


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session



data_path ="/home/ubuntu/draw/data/"
train_path = data_path+"train/"
num_classes = 340 




classfiles = glob(train_path+"*.csv")
nametoid = {v[:-4].split("/")[-1].replace(" ", "_") :i
							 for i, v in enumerate(classfiles) if "df_all_raw" not in v} #adds underscores
idtoname = {i: v[:-4].split("/")[-1].replace(" ", "_")
							 for i, v in enumerate(classfiles) if "df_all_raw" not in v} #adds underscores

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
									# (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras




def batch_relu(x):
	x = BatchNormalization()(x)    
	x = Activation('relu')(x)
	
	return x


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


def get_model_with_imagenet_weights(imgnet_model, freeze=True):
	base_model = imgnet_model
	# add a global spatial average pooling layer
	x = base_model.output
	# x = GlobalMaxPooling2D()(x)
	# let's add a fully-connected layer
	x = Dense(512, activation='relu')(x)
	x = Dropout(0.5)(x)
	predictions = Dense(num_classes, activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	if freeze:
		for layer in base_model.layers:
			layer.trainable = False	
			
	print(model.summary())

	return model


def load_sample_set():
	print ("loading sample data based on fold8, nonrec..")
	samplel_df_fold8 = pd.read_csv("../data/sample_df_nonrec_fold8.csv")
	valid_df_fold8 = pd.read_csv("../data/valid_df_nonrec_fold8.csv")

	print("training set size:",samplel_df_fold8.shape)
	return samplel_df_fold8, valid_df_fold8



def load_train_valid(fold):
	print ("loading data..")

	df_all = pd.read_csv(data_path+"df_all_nonrec.csv")

	_10fold0 = load_obj("../data/folds10_nonrec.pik")[fold]
	train_index = _10fold0["train_index"]
	valid_index = _10fold0["test_index"]
	# valid_df = df_all.iloc[valid_index] #pd.read_csv("../data/valid_df.csv")
	# train_df = df_all.iloc[train_index]
	# del df_all
	# gc.collect()
	return df_all.iloc[train_index], df_all.iloc[valid_index]



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


	# mnet = GlobalMaxPooling2D()(mnet_raw.output)
	# mnet = Dense(256, activation="relu")(mnet)
	# mnet = Dropout(0.3)(mnet)
	# mnet = Dense(num_classes,activation="softmax")(mnet)

	# model = Model(inputs = mnet_raw.input, outputs = mnet)	

	# for layer in mnet_raw.layers:
	#     layer.trainable = False	
