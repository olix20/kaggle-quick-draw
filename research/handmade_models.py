from utils import *




def get_multibranch_inception(imsize):
	max_length = 196
	points_input = Input(shape=(max_length,3),  name='points_input')
	img_input = Input(shape=(imsize,imsize,3),  name='img_input')



	rnn = BatchNormalization()(points_input)

	rnn = Conv1D(48,5,activation='relu')(rnn)
	rnn = Conv1D(64,5,activation='relu')(rnn)
	rnn = Dropout(0.3)(rnn)
	rnn = Conv1D(96,5,activation='relu')(rnn)
	rnn = Dropout(0.3)(rnn)

	rnn = Bidirectional(CuDNNLSTM(128, return_sequences = False))(rnn) #CuDNNLSTM
	rnn = Dropout(0.3)(rnn)

	cnn_raw = keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=(imsize, imsize, 3), include_top=False,
	  weights="imagenet", classes=num_classes)

	cnn = GlobalMaxPooling2D()(cnn_raw(img_input))
	cnn = Dense(256, activation="relu")(cnn)
	cnn = Dropout(0.3)(cnn)


	output = concatenate([cnn, rnn])
	# output = Dropout(0.3)(output)
	output = Dense(512, activation="relu")(output)
	output = Dropout(0.3)(output)

	output = Dense(num_classes,activation="softmax")(output)

	model = Model(inputs = [points_input,img_input], outputs = output)	
	print(model.summary())

	return model 	



def get_simple_crnn_1d():
	max_length = 196
	points_input = Input(shape=(max_length,3),  name='points_input')

	rnn = Conv1D(32,15,activation='relu')(points_input)
	rnn = Conv1D(32,15,activation='relu')(rnn)

	rnn = MaxPool1D(pool_size=2,strides=1)(rnn)

	rnn = Conv1D(64,5,activation='relu')(rnn)
	rnn = Conv1D(64,5,activation='relu')(rnn)

	rnn = MaxPool1D(pool_size=2)(rnn)

	rnn = Conv1D(128,5,activation='relu')(rnn)
	rnn = Conv1D(128,5,activation='relu')(rnn)

	rnn = MaxPool1D(pool_size=2)(rnn)

	rnn = Conv1D(128,5,activation='relu')(rnn)
	rnn = Conv1D(128,5,activation='relu')(rnn)

	rnn = Bidirectional(CuDNNLSTM(128))(rnn) #CuDNNLSTM

	rnn = Dropout(0.3)(rnn)
	rnn = Dense(512)(rnn)
	rnn = Dropout(0.3)(rnn)

	rnn = Dense(num_classes, activation="softmax")(rnn)

	rnn = Model(inputs = points_input, outputs = rnn)	
	print(rnn.summary())

	return rnn


def get_mader_crnn_1d():
	max_length = 196
	points_input = Input(shape=(max_length,3),  name='points_input')

	rnn = BatchNormalization()(points_input)

	rnn = Conv1D(48,5,activation='relu')(rnn)
	rnn = Conv1D(64,5,activation='relu')(rnn)
	rnn = Dropout(0.3)(rnn)
	rnn = Conv1D(96,5,activation='relu')(rnn)
	rnn = Dropout(0.3)(rnn)

	rnn = Bidirectional(CuDNNLSTM(128, return_sequences = True))(rnn) #CuDNNLSTM
	rnn = Dropout(0.3)(rnn)
	rnn = Bidirectional(CuDNNLSTM(128, return_sequences = False))(rnn) #CuDNNLSTM
	rnn = Dropout(0.3)(rnn)
	rnn = Dense(512)(rnn)
	rnn = Dropout(0.3)(rnn)

	rnn = Dense(num_classes,activation="softmax")(rnn)

	rnn = Model(inputs = points_input, outputs = rnn)	
	print(rnn.summary())

	return rnn	

def get_mader_crnn_1d_modified():
	max_length = 196
	points_input = Input(shape=(max_length,3),  name='points_input')

	rnn = BatchNormalization()(points_input)

	rnn = Conv1D(64,7,activation='relu')(rnn)
	rnn = Conv1D(64,7,activation='relu')(rnn)
	rnn = Dropout(0.3)(rnn)
	rnn = Conv1D(128,7,activation='relu')(rnn)
	rnn = Dropout(0.3)(rnn)

	rnn = Bidirectional(CuDNNLSTM(128, return_sequences = False))(rnn) #CuDNNLSTM
	rnn = Dropout(0.3)(rnn)
	# rnn = Bidirectional(CuDNNLSTM(128, return_sequences = False))(rnn) #CuDNNLSTM
	# rnn = Dropout(0.3)(rnn)
	rnn = Dense(512)(rnn)
	rnn = Dropout(0.3)(rnn)

	rnn = Dense(num_classes,activation="softmax")(rnn)

	rnn = Model(inputs = points_input, outputs = rnn)	
	print(rnn.summary())

	return rnn	

def get_multi_branch_mobilenet_crnn1d(imsize):
	max_length = 196
	points_input = Input(shape=(max_length,3),  name='points_input')
	img_input = Input(shape=(imsize,imsize,1),  name='img_input')



	rnn = BatchNormalization()(points_input)

	rnn = Conv1D(48,5,activation='relu')(rnn)
	rnn = Conv1D(64,5,activation='relu')(rnn)
	rnn = Dropout(0.3)(rnn)
	rnn = Conv1D(96,5,activation='relu')(rnn)
	rnn = Dropout(0.3)(rnn)

	rnn = Bidirectional(CuDNNLSTM(128, return_sequences = True))(rnn) #CuDNNLSTM
	rnn = Dropout(0.3)(rnn)
	rnn = Bidirectional(CuDNNLSTM(128, return_sequences = False))(rnn) #CuDNNLSTM
	# rnn = Dropout(0.3)(rnn)
	# rnn = Dense(512)(rnn)
	# rnn = Dropout(0.3)(rnn)



	mnet = MobileNetV2(input_shape=(imsize, imsize, 1),  weights=None, classes=num_classes, include_top=False)
	mnet = mnet(img_input)
	mnet = GlobalAveragePooling2D()(mnet)

	output = concatenate([mnet, rnn])
	output = Dropout(0.3)(output)
	output = Dense(512)(output)
	output = Dropout(0.3)(output)

	output = Dense(num_classes,activation="softmax")(output)

	model = Model(inputs = [points_input,img_input], outputs = output)	
	print(model.summary())

	return model 






def get_sketchnet_cnn(imsize):

	img_input = Input(shape=(imsize, imsize, 1),  name='img_input')

	# x = BatchNormalization()(img_input)

	x = Conv2D(64, kernel_size=(15, 15), strides=3,  padding='same', activation='relu')(img_input)
	x = MaxPooling2D((3,3),strides=2)(x)

	x = Conv2D(128, kernel_size=(5, 5), strides=1 , padding='same', activation='relu')(x)
	x = MaxPooling2D((3,3),strides=2)(x)

	x = Conv2D(256, kernel_size=(3, 3),strides=1, padding='same', activation='relu')(x)
	x = Conv2D(256, kernel_size=(3, 3),strides=1, padding='same', activation='relu')(x)
	x = Conv2D(256, kernel_size=(3, 3),strides=1, padding='same', activation='relu')(x)
	x = MaxPooling2D((3,3),strides=2)(x)

	x = Conv2D(512, kernel_size=(7, 7),strides=1, padding='same', activation='relu')(x)
	x = Conv2D(512, kernel_size=(1, 1),strides=1, padding='same', activation='relu')(x)
	# x = Dropout(0.3)(x)

	# x = Conv2D(num_classes, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(x)
	x = GlobalAveragePooling2D()(x)
	# x = Activation('softmax')(x)

	x = Dense(512)(x)
	x = Dropout(0.3)(x)

	x = Dense(num_classes,activation="softmax")(x)

	model = Model(inputs = img_input, output = x)
	print(model.summary())
	return model 



def get_simple_cnn2d(imsize):

	img_input = Input(shape=(imsize, imsize, 1),  name='img_input')

	x = BatchNormalization()(img_input)

	x = Conv2D(32, kernel_size=(5, 5),  padding='same', activation='relu')(img_input)	
	x = Conv2D(64, kernel_size=(5, 5),  padding='same', activation='relu')(x)
	x = Conv2D(128, kernel_size=(5, 5),  padding='same', activation='relu')(x)

	x = GlobalAveragePooling2D()(x)
	# x = Activation('softmax')(x)

	x = Dense(512)(x)
	x = Dropout(0.3)(x)

	x = Dense(num_classes,activation="softmax")(x)

	model = Model(inputs = img_input, output = x)
	print(model.summary())
	return model 


def get_crnn_2d(imsize):

	max_stroke_sequence = 20 
	strokes_input = Input(shape=(imsize, imsize* max_stroke_sequence, 1),  name='strokes_input')

	x = Conv2D(64,(15,15),padding='same')(strokes_input)
	x = batch_relu(x)

	x = MaxPooling2D((2,5))(x)

	x = Conv2D(64,(7,7),padding='same')(x)
	x = batch_relu(x)

	x = MaxPooling2D((2,7))(x)


	x = Conv2D(64,(5,5),padding='same')(x)
	x = batch_relu(x)

	x = MaxPooling2D((2,7))(x)

	print (int(x.shape[-1]) * int(x.shape[-2]))
	x = Reshape((int(x.shape[-3]) ,int(x.shape[-1]) * int(x.shape[-2])))(x)

	x = Bidirectional(CuDNNLSTM(64,return_sequences=False))(x)
	x = Dense(num_classes,activation="softmax")(x)
	
	model = Model(inputs =strokes_input, output = x)

	print(model.summary())
	return model 


def get_crnn_2d_discrete(imsize):

	stroke_inputs = []
	max_stroke_sequence = 20 
	for i in range(max_stroke_sequence):
		strokes_input = Input(shape=(imsize, imsize , 1),  name=f'strokes_input_{i}')
		stroke_inputs.append(strokes_input)
	
	mnet = MobileNetV2(input_shape=(imsize, imsize, 1),  weights=None, classes=num_classes, include_top=False)

	mnet_outpus = []
	for i in range(max_stroke_sequence):
		mnet_output = mnet(stroke_inputs[i])
		mnet_output = GlobalMaxPooling2D()(mnet_output)
		mnet_output = keras.backend.expand_dims(mnet_output,axis=0)
		mnet_outpus.append(mnet_output)

	x = concatenate(mnet_outpus)
	x = Dropout(0.3)(x)
	# print (int(x.shape[-1]) * int(x.shape[-2]))
	# x = Reshape((int(x.shape[-3]) ,int(x.shape[-1]) * int(x.shape[-2])))(x)

	x = Bidirectional(CuDNNLSTM(128,return_sequences=False))(x)
	x = Dropout(0.3)(x)

	x = Dense(512)(x)
	x = Dropout(0.3)(x)

	x = Dense(num_classes,activation="softmax")(x)
	
	model = Model(inputs =stroke_inputs, output = x)

	print(model.summary())
	return model 


def get_crnn_2d_distributed(imsize):
	max_stroke_sequence = 20 
	input_sequences = Input(shape=(max_stroke_sequence, imsize, imsize, 1))

	# mnet_input = Input(shape=(imsize, imsize, 1))
	mnet_raw = MobileNetV2(input_shape=(imsize, imsize, 1),  weights=None, classes=num_classes, include_top=False)
	mnet = GlobalMaxPooling2D()(mnet_raw.output)
	mnet = Dense(128, activation="relu")(mnet)
	mnet = Model(inputs=mnet_raw.input, outputs = mnet)


	x = TimeDistributed(mnet)(input_sequences)
	# x = TimeDistributed(GlobalMaxPooling2D())
	# print (x.shape)
	# x = Reshape((max_stroke_sequence, int(x.shape[-1]) * int(x.shape[-2]))) (x)	
	# x = Dropout(0.3)(x)
	# print (int(x.shape[-1]) * int(x.shape[-2]))
	# x = Reshape((int(x.shape[-3]) ,int(x.shape[-1]) * int(x.shape[-2])))(x)

	x = Bidirectional(CuDNNLSTM(128,return_sequences=False))(x)
	x = Dropout(0.3)(x)

	x = Dense(512, activation="relu")(x)
	x = Dropout(0.3)(x)

	x = Dense(num_classes,activation="softmax")(x)
	
	model = Model(inputs =input_sequences, output = x)

	print(model.summary())
	return model 




def multibranch_test():

	imsize=64  
	batch_size=128
	fold = 8 #np.random.choice( range(1,10))

	model = get_multi_branch_mobilenet_crnn1d(imsize)
	name = "mobnet_crnn1d_multibranch"
	exp_name = f"{name}_im{imsize}_fold{fold}_nonrec_" + "noaug"

	train_df, valid_df = load_sample_set()

	print(f"Beginning training for {exp_name}")
	exp = Experiment_Sample(imsize, batch_size, exp_name, data_generator=RNN_1d_generator)	
	exp.train(model,train_df,valid_df,continue_training = False, do_aug=False)		



def crnn1d_test():
	imsize=64  
	batch_size=128
	fold = 8 #np.random.choice( range(1,10))
	train_df, valid_df = load_sample_set()

	model = get_mader_crnn_1d_modified()
	name = "mader_mofified_sample"
	exp_name = f"{name}_im{imsize}_fold{fold}_nonrec_" + "noaug"

	print(f"Beginning training for {exp_name}")
	exp = Experiment_Sample(imsize, batch_size, exp_name, data_generator=RNN_1d_generator)	
	exp.train(model,train_df,valid_df,continue_training = False, do_aug=False)		



def crnn2d_test():
	imsize=64  
	batch_size=64
	fold = 8 #np.random.choice( range(1,10))

	model = get_crnn_2d_distributed(imsize)
	train_df, valid_df = load_sample_set()

	name = "discrete_crnn2d_bn128_sample"
	exp_name = f"{name}_im{imsize}_fold{fold}_nonrec_" + "noaug"

	print(f"Beginning training for {exp_name}")
	exp = Experiment_Sample(imsize, batch_size, exp_name, data_generator=CRNN_2d_generator)	
	exp.train(model,train_df,valid_df, continue_training=False, do_aug=False)		




if __name__ == '__main__':
	crnn2d_test()