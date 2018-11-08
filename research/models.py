from utils import *

def get_mnet_pretrained(imsize, freeze=True):
	mnet_raw = MobileNetV2(
		input_shape=(imsize, imsize, 3), alpha=1.,  weights="imagenet",
		pooling="max", classes=num_classes, include_top=False)

	model = get_model_with_imagenet_weights(mnet_raw, freeze)
	preprocess_input = keras.applications.mobilenet_v2.preprocess_input

	return model , preprocess_input


def get_incresnetv2_imagenet(imsize, freeze=True):
	raw = keras.applications.inception_resnet_v2.InceptionResNetV2(
		input_shape=(imsize, imsize, 3), include_top=False,
	  weights="imagenet", pooling="max",  classes=num_classes)
	
	model = get_model_with_imagenet_weights(raw, freeze)
	preprocess_input = keras.applications.inception_resnet_v2.preprocess_input

	return model, preprocess_input


def get_densenet_121_imagenet(imsize, freeze=True):
	densenet_raw = keras.applications.densenet.DenseNet121(input_shape=(imsize, imsize, 3), 
		include_top=False, pooling="max", 
	  weights="imagenet", classes=num_classes)

	model = get_model_with_imagenet_weights(densenet_raw, freeze)
	preprocess_input = keras.applications.densenet.preprocess_input

	return model, preprocess_input


def get_incresnetv2_imagenet(imsize, freeze=True):
	raw = keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=(imsize, imsize, 3), include_top=False,
	  weights="imagenet", classes=num_classes)

	model = get_model_with_imagenet_weights(raw, freeze)
	preprocess_input = keras.applications.inception_resnet_v2.preprocess_input

	return model, preprocess_input


def get_multi_branch_mobilenet_crnn1d(imsize):
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
	# rnn = Bidirectional(CuDNNLSTM(128, return_sequences = False))(rnn) #CuDNNLSTM
	# rnn = Dropout(0.3)(rnn)
	# rnn = Dense(512)(rnn)
	# rnn = Dropout(0.3)(rnn)
	mnet_raw = MobileNetV2(input_shape=(imsize, imsize, 3), alpha=1.,  weights="imagenet", classes=num_classes, include_top=False)

	mnet = GlobalMaxPooling2D()(mnet_raw(img_input))
	mnet = Dense(256, activation="relu")(mnet)
	mnet = Dropout(0.3)(mnet)

	output = concatenate([mnet, rnn])
	# output = Dropout(0.3)(output)
	output = Dense(512, activation="relu")(output)
	output = Dropout(0.3)(output)
	output = Dense(num_classes,activation="softmax")(output)

	model = Model(inputs = [points_input,img_input], outputs = output)	
	print(model.summary())


	return model ,  keras.applications.inception_resnet_v2.preprocess_input


