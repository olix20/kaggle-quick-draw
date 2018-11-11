from utils import *
from experiments import *
from qd_data import *
from models import *


def load_super_small_set():
	train_df = pd.read_csv("../data/small-set.csv")
	valid_df = train_df

	return 	train_df, valid_df

def cnn2d_3channel_test():
	imsize=96	  
	batch_size=128
	# fold = 8 #np.random.choice( range(1,10))

	name = "xception_real_3d"
	exp_name = f"{name}_im{imsize}_batch{batch_size}"

	# model = get_densenet_121_imagenet(imsize)
	# model = get_incresnetv2_imagenet(imsize)
	# model = get_multi_branch_mobilenet_crnn1d(imsize)
	# get_resnet_imagenet
	model, preprocess_input = get_exception_3d(imsize)
	del model
	# train_df, valid_df = load_super_small_set()
	# train_df, valid_df = load_sample_set()
	# train_df = train_df.iloc[:len(train_df)//10]

	print(f"Beginning training for {exp_name}")

	# exp = Experiment_Sample(imsize, batch_size, exp_name,
	#  data_generator=CNN2D_generator, 
	#  preprocess_input=preprocess_input, 
	#  num_workers=os.cpu_count(),
	#  min_lr=1e-4) 
	
	exp = GZ_experiment(imsize, batch_size, exp_name,
	 preprocess_input=preprocess_input, 
	 num_workers=os.cpu_count(),
	 min_lr=1e-4) 

	exp.train(None,continue_training=True , lr=2e-3)



if __name__ == '__main__':
	cnn2d_3channel_test()

#CRNN_2d_generator) #QD_Datagen_Sample)		
