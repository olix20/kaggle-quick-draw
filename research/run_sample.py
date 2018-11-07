from utils import *
from experiments import *
from qd_data import *
from augmentations import * 
from models import *



def cnn2d_3channel_test():
	imsize=96	  
	batch_size=128
	fold = 8 #np.random.choice( range(1,10))


	name = "mnet_sample_rgb_lr2e-4"
	exp_name = f"{name}_im{imsize}_fold{fold}_batch{batch_size}_nonrec"

	# model = get_densenet_121_imagenet(imsize)
	# model = get_incresnetv2_imagenet(imsize)
	# model = get_multi_branch_mobilenet_crnn1d(imsize)
	model, preprocess_input = get_mnet_pretrained(imsize)
	train_df, valid_df = load_sample_set()


	print(f"Beginning training for {exp_name}")
	exp = Experiment_Sample(imsize, batch_size, exp_name, data_generator=CNN2D_generator) 
	# densenet
	exp.train(model,train_df,valid_df, continue_training=False, 
		preprocess_input=preprocess_input, num_workers=os.cpu_count())


if __name__ == '__main__':
	cnn2d_3channel_test()

#CRNN_2d_generator) #QD_Datagen_Sample)		