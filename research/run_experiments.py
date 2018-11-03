#!/usr/bin/env python
# coding: utf-8

from utils import *
from models import *
from qd_data import *
from augmentations import *


import numpy as np
import gc
import keras

from keras.applications.mobilenet_v2 import MobileNetV2
from keras.optimizers import Adam


classfiles = glob(train_path+"*.csv")
nametoid = {v[:-4].split("/")[-1].replace(" ", "_") :i
							 for i, v in enumerate(classfiles) if "df_all_raw" not in v} #adds underscores
idtoname = {i: v[:-4].split("/")[-1].replace(" ", "_")
							 for i, v in enumerate(classfiles) if "df_all_raw" not in v} #adds underscores


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


def run_model(exp_name, model, train_df, valid_df,imsize, batch_size):
	print(f"Beginning training for {exp_name}")
	exp = Experiment(imsize, batch_size, exp_name)	
	exp.train(model,train_df,valid_df,continue_training=True,do_aug=False)
	exp.predict()
	del exp


def exp_im128():
	imsize=128  
	batch_size=128
	fold = 8 #np.random.choice( range(1,10))
	train_df, valid_df = load_train_valid(fold)

	# ("resnet50",keras.applications.resnet50.ResNet50(input_shape=(imsize, imsize, 1), weights=None, classes=num_classes)),
	# ("inception_resnet_v2",keras.applications.InceptionResNetV2(input_shape=(imsize, imsize, 1),  weights=None, classes=num_classes))
	# ("xception",keras.applications.xception.Xception(input_shape=(imsize, imsize, 1),  weights=None, classes=num_classes))
	# ("resnet50_augv2",keras.applications.resnet50.ResNet50(input_shape=(imsize, imsize, 1), weights=None, classes=num_classes))
	# ("inception_resnet_v2_augv2",keras.applications.InceptionResNetV2(input_shape=(imsize, imsize, 1),  weights=None, classes=num_classes))
	# ("MobileNetV2_augv2", MobileNetV2(input_shape=(imsize, imsize, 1),  weights=None, classes=num_classes))
# ("DenseNet169_augv2",keras.applications.densenet.DenseNet169(input_shape=(imsize, imsize, 1),  weights=None, classes=num_classes))	
	models = [("MobileNetV2_noaug", MobileNetV2(input_shape=(imsize, imsize, 1),  weights=None, classes=num_classes))]
	
	for (name, model) in models:
		exp_name = f"{name}_im{imsize}_fold{fold}_nonrec"
		run_model(exp_name, model,train_df,valid_df,imsize,batch_size)		

if __name__ == '__main__':
	exp_im128()
