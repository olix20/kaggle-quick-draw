from utils import *
from models import *
from qd_data import *

import numpy as np
import gc
import keras

from keras.applications.mobilenet_v2 import MobileNetV2
from keras.optimizers import Adam




def aug_test():
	imsize=64  
	batch_size=128
	fold = 8 #np.random.choice( range(1,10))
	train_df, valid_df = load_sample_set()


	model = MobileNetV2(input_shape=(imsize, imsize, 1),  weights=None, classes=num_classes)
	name = "MobileNetV2_noweight_sample"

	exp_name = f"{name}_im{imsize}_fold{fold}_nonrec_rerun256" + "iaarot10"
	print(f"Beginning training for {exp_name}")
	exp = Experiment_Sample(imsize, batch_size, exp_name)	
	exp.train(model,train_df,valid_df, stroke_aug_fn=None, iaa_seq=rotation10(),do_aug=True)		




def all_iaa_augs():
	sometimes = lambda aug: iaa.Sometimes(0.5, aug)
	seq = iaa.Sequential([
		iaa.Fliplr(0.5),
		iaa.Flipud(0.2),
		sometimes(iaa.Affine(scale={"y": (0.8, 1.2)} ,  cval=1)),
		sometimes(iaa.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},cval=1)),
		sometimes(iaa.Affine(shear=(-5, 5),  cval=1)),
		sometimes(iaa.Affine(rotate=(-10, 10),  cval=1))
		])				
	return seq

def rotate_and_translate():
	sometimes = lambda aug: iaa.Sometimes(0.5, aug)
	seq = iaa.Sequential([
	    iaa.Fliplr(0.5),
	#     iaa.Flipud(0.2),
	#     sometimes(iaa.Affine(scale={"y": (0.8, 1.2)} ,  cval=1)),
	    sometimes(iaa.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},cval=1)),
	#     sometimes(iaa.Affine(shear=(-5, 5),  cval=1)),
	    sometimes(iaa.Affine(rotate=(-10, 10),  cval=1))
	    ])	

	return seq

def random_vertex_and_stroke_drop(strokes, prob=0.1):
    augmented = []
    strokes = ast.literal_eval(strokes)
    
    for i,stroke in enumerate(strokes):
        
        stroke_drop_prob = prob*((i+1)/len(strokes))
        # drop final strokes with a higher chance 
        if np.random.rand() < stroke_drop_prob:
            # print("dropping stroke")
            continue
            
        if len(stroke[0]) < 3 : 
            augmented.append(stroke)
            continue
        else:
            new_stroke =[[],[]]
            for i in range(0,len(stroke[0])):
                if i > 0 and i < len(stroke[0])-1:
                    if np.random.rand() < prob:
                        continue 
                        
                new_stroke[0].append(stroke[0][i])
                new_stroke[1].append(stroke[1][i])
            
            augmented.append(new_stroke)
    return str(augmented)


def random_vertex_drop_and_noise(strokes, prob=0.1):
	augmented = []
	for stroke in ast.literal_eval(strokes):
		if len(stroke[0]) < 3 : 
			augmented.append(stroke)
			continue
		else:
			new_stroke =[[],[]]
			for i in range(0,len(stroke[0])):
				if i > 0 and i < len(stroke[0])-1:
					if np.random.rand() < prob:
						# print("skipping adding item")
						continue 
					if np.random.rand() < prob:
						# print("adjusting stroke by a random number")
						delta_x = np.random.randint(-5,5)
						delta_y = np.random.randint(-5,5)
						stroke[0][i] += delta_x
						stroke[1][i] += delta_y
						
				new_stroke[0].append(stroke[0][i])
				new_stroke[1].append(stroke[1][i])
			
			augmented.append(new_stroke)
	return str(augmented)



def random_vertex_drop(strokes, prob=0.1):
	augmented = []
	for stroke in ast.literal_eval(strokes):
		if len(stroke[0]) < 3 : 
			augmented.append(stroke)
			continue
		else:
			new_stroke =[[],[]]
			for i in range(0,len(stroke[0])):
				if i > 0 and i < len(stroke[0])-1:
					if np.random.rand() < prob:
						# print("skipping adding item")
						continue 
					# if np.random.rand() < prob:
					# 	# print("adjusting stroke by a random number")
					# 	delta_x = np.random.randint(-5,5)
					# 	delta_y = np.random.randint(-5,5)
					# 	stroke[0][i] += delta_x
					# 	stroke[1][i] += delta_y
						
				new_stroke[0].append(stroke[0][i])
				new_stroke[1].append(stroke[1][i])
			
			augmented.append(new_stroke)
	return str(augmented)




def random_vertex_noise(strokes, prob=0.1):
	augmented = []
	for stroke in ast.literal_eval(strokes):
		if len(stroke[0]) < 3 : 
			augmented.append(stroke)
			continue
		else:
			new_stroke =[[],[]]
			for i in range(0,len(stroke[0])):
				if i > 0 and i < len(stroke[0])-1:
					# if np.random.rand() < prob:
					# 	# print("skipping adding item")
					# 	continue 
					if np.random.rand() < prob:
						# print("adjusting stroke by a random number")
						delta_x = np.random.randint(-5,5)
						delta_y = np.random.randint(-5,5)
						stroke[0][i] += delta_x
						stroke[1][i] += delta_y
						
				new_stroke[0].append(stroke[0][i])
				new_stroke[1].append(stroke[1][i])
			
			augmented.append(new_stroke)
	return str(augmented)



def simple_hr_flip():
	return iaa.Sequential([
	iaa.Fliplr(0.5)
	])

def simple_vertical_flip():
	return iaa.Sequential([
	iaa.Flipud(0.5)
	])


def rotation10():
	sometimes = lambda aug: iaa.Sometimes(0.5, aug)

	return iaa.Sequential([
	sometimes(iaa.Affine(
	# translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -10 to +10 percent (per axis)
		rotate=(-10, 10), # rotate by -45 to +45 degrees
#         shear=(-1, 1), # shear by -16 to +16 degrees
		cval=1
	))   	

	])	


def shear5():
	sometimes = lambda aug: iaa.Sometimes(0.5, aug)

	return iaa.Sequential([
	sometimes(iaa.Affine(
	# translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -10 to +10 percent (per axis)
		# rotate=(-10, 10), # rotate by -45 to +45 degrees
		shear=(-5, 5), # shear by -16 to +16 degrees
		cval=1
	))   	

	])	

def translate05():
	sometimes = lambda aug: iaa.Sometimes(0.5, aug)

	return iaa.Sequential([
	sometimes(iaa.Affine(
	translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, # translate by -10 to +10 percent (per axis)
		# rotate=(-10, 10), # rotate by -45 to +45 degrees
		# shear=(-5, 5), # shear by -16 to +16 degrees
		cval=1
	))   	

	])	


def zoom05():
	sometimes = lambda aug: iaa.Sometimes(0.5, aug)

	return iaa.Sequential([
	sometimes(iaa.Affine(scale={"y": (0.8, 1.2)} ,      # rotate=(-10, 10), # rotate by -45 to +45 degrees
			cval=1
	))   	

	])	

def localaffine():
	sometimes = lambda aug: iaa.Sometimes(0.5, aug)

	return iaa.Sequential([
	sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05), # sometimes move parts of the image around
			cval=1
	))   	
	])	





if __name__ == '__main__':
	aug_test()