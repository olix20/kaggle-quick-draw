#!/usr/bin/env python
# coding: utf-8

from utils import *
from experiments import *
from qd_data import *
from augmentations import * 
from models import *

	
def run_experiment():

	imsize=96  
	batch_size=64
	fold = 9 #np.random.choice( range(1,10))
#CRNN_2d_generator) #QD_Datagen_Sample)	

	name = "something"
	exp_name = f"{name}_im{imsize}_batch_{batch_size}_fold{fold}_nonrec"

	model, preprocess_input = get_mnet_pretrained(imsize)

	exp = Experiment_Sample(
		imsize, batch_size, exp_name, 
		data_generator=CNN2D_generator, 
		preprocess_input=preprocess_input) 

	train_df, valid_df = load_train_valid(fold)
	print(f"Beginning training for {exp_name}")

	exp.train(model,train_df,valid_df, continue_training=True)		
	
	# exp.predict()


if __name__ == '__main__':
	run_experiment()


	# ("resnet50",keras.applications.resnet50.ResNet50(input_shape=(imsize, imsize, 1), weights=None, classes=num_classes)),
	# ("inception_resnet_v2",keras.applications.InceptionResNetV2(input_shape=(imsize, imsize, 1),  weights=None, classes=num_classes))
	# ("xception",keras.applications.xception.Xception(input_shape=(imsize, imsize, 1),  weights=None, classes=num_classes))
	# ("resnet50_augv2",keras.applications.resnet50.ResNet50(input_shape=(imsize, imsize, 1), weights=None, classes=num_classes))
	# ("inception_resnet_v2_augv2",keras.applications.InceptionResNetV2(input_shape=(imsize, imsize, 1),  weights=None, classes=num_classes))
	# ("MobileNetV2_augv2", MobileNetV2(input_shape=(imsize, imsize, 1),  weights=None, classes=num_classes))
# ("DenseNet169_augv2",keras.applications.densenet.DenseNet169(input_shape=(imsize, imsize, 1),  weights=None, classes=num_classes))	
	# models = [("multibranch_inception_noaug", None)] #get_multibranch_inception(imsize))]
	# models = [("inception_imagenet_noaug", None)]#get_incresnetv2_imagenet(imsize))] 
	# models  = [("mnet_imagenet", None)]#get_mnet_pretrained(imsize))]




# def run_model(exp_name, model, train_df, valid_df,imsize, batch_size):
# 	print(f"Beginning training for {exp_name}")

# 	exp = Experiment_Sample(imsize, batch_size, 
# 	exp_name, data_generator=CNN2D_generator, preprocess_input=keras.applications.mobilenet_v2.preprocess_input) #CRNN_2d_generator) #QD_Datagen_Sample)	
# 	# densenet
# 	exp.train(model,train_df,valid_df, continue_training=True, do_aug=False, 
# 		)		
	
# 	# exp = Experiment(imsize, batch_size, exp_name)	
# 	# exp.train(model,train_df,valid_df,continue_training=True,do_aug=False)
# 	exp.predict()
# 	del exp


# def exp_im128():
# 	imsize=96  
# 	batch_size=64
# 	fold = 9 #np.random.choice( range(1,10))


# 	train_df, valid_df = load_train_valid(fold)
	
# 	for (name, model) in models:
# 		exp_name = f"{name}_im{imsize}_batch_{batch_size}_fold{fold}_nonrec"
# 		run_model(exp_name, model,train_df,valid_df,imsize,batch_size)	