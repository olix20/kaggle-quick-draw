from utils import *
from qd_data import *

DP_DIR="../data/csv_gz"


class GZ_experiment:
	def __init__(self, 
				 imsize, 
				 batch_size, 
				 exp_name,
				 preprocess_input=None,
				 min_lr=2e-4,
				 num_workers = None 
				 ):
		
		self.imsize = imsize
		self.batch_size = batch_size
		self.exp_name = exp_name     
		self.preprocess_input = preprocess_input
		self.min_lr = min_lr
		self.num_workers = num_workers or os.cpu_count()


	def train(self, 
			  model,
			 continue_training=True,
			 do_aug=False, 
			 lr=None):
		
		self.model = model


		print("loading valid df .. ")
		valid_df = pd.read_csv(os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)  ), nrows=140000) #
		x_valid = df_to_image_array_xd(valid_df, self.imsize,preprocess_input=self.preprocess_input)
		y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)
		print(x_valid.shape, y_valid.shape)
		print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 1024.**3 ))

		
		# continue training from a broken run, this loads last state of optimizer too!
		last_model = sorted(glob(f"/home/ubuntu/draw/weights/{self.exp_name}*.hdf5"), 
			key=os.path.getmtime,reverse=True)		

		if continue_training and len(last_model) > 0:
			# appending a token to indicate continued experiment
			self.exp_name = self.exp_name + "x"*len(last_model)

			last_model = last_model[0]
			self.model = keras.models.load_model(last_model,
												 custom_objects={"top_3_accuracy":self.top_3_accuracy})   
		
			print (f"Found existing model: {last_model}")
			print(f"Continuing training, new exp_name: {self.exp_name}")

		else:
			# important to put this under else condition otherwise learning rate will be reset
			self.model.compile(loss='categorical_crossentropy',
			  optimizer=Adam(lr=2e-3),
			  metrics=['accuracy', self.top_3_accuracy])   			

		# override lr if specified
		if lr:
			self.model.compile(loss='categorical_crossentropy',
			  optimizer=Adam(lr=lr),
			  metrics=['accuracy', self.top_3_accuracy]) 


		callbacks = self.get_callbacks(self.exp_name, self.batch_size)

		train_datagen = image_generator_xd(size=self.imsize, 
			batchsize=self.batch_size, ks=range(NCSVS - 1),preprocess_input=self.preprocess_input)
		self.model.fit_generator(train_datagen,
			 steps_per_epoch=4096,
			 epochs = 10000,
			 validation_data = (x_valid, y_valid),
			 callbacks = callbacks,
			 verbose = 1, 
			 workers=1, #self.num_workers, 
			 max_queue_size=100,
			 use_multiprocessing=False) 
		

	def top_3_accuracy(self, x,y): 
		t3 = top_k_categorical_accuracy(x,y, 3)
		return t3

	def get_callbacks(self, exp_name, batch_size):
		reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, cooldown=3, verbose=1,
										   mode='auto', min_delta=0.005, min_lr=self.min_lr)
		earlystop = EarlyStopping(monitor='val_top_3_accuracy', mode='max', patience=8) 
		# earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=5) 

		exp_path = f'/home/ubuntu/draw/weights/{exp_name}.hdf5'
		save_best = ModelCheckpoint(monitor='val_loss',
								 filepath=exp_path, #+'-{val_loss:.2f}.hdf5',
								 save_best_only=True,
								 save_weights_only=False,
								 mode='min')
		csv_logger = CSVLogger('/home/ubuntu/draw/logs/csv/{}_train_log.csv'.format(exp_name))

		callbacks = [reduceLROnPlat, earlystop, save_best,csv_logger]
		return callbacks


	def predict(self):
		last_model = sorted(glob(f"/home/ubuntu/draw/weights/{self.exp_name}*.hdf5"), 
							  key=os.path.getmtime,reverse=True)   
		if len(last_model) > 0:
			last_model=last_model[0]
		else:
			print("Prediction error: model doesn't exist for the given exp_name")
			return -1
		
		print (f"Predicting using model found in {last_model} ..")
		model = keras.models.load_model(last_model,
											 custom_objects={"top_3_accuracy":self.top_3_accuracy})        


		test_df = pd.read_csv('/home/ubuntu/draw/data/test_simplified.csv')
		test_gen = TestGenerator(test_df, self.batch_size, imsize=self.imsize, 
													 preprocess_input=self.preprocess_input)     
		
		test_predictions = model.predict_generator(test_gen, 
												   steps=len(test_gen), 
												   verbose=1)
		test_predictions = test_predictions[:test_df.shape[0]]
		np.save(f"/home/ubuntu/draw/predictions/{self.exp_name}.npy",test_predictions)
		



class GZ_experiment_1d(GZ_experiment):

	def train(self, 
			  model,
			 continue_training=True,
			 lr=None):
		
		self.model = model


		print("loading valid df .. ")
		valid_df = pd.read_csv(os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=80000) # 
		x_valid = df_to_image_array_1d(valid_df, self.imsize,preprocess_input=self.preprocess_input)
		y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)
		print(x_valid.shape, y_valid.shape)
		print('Validation array memory {:.2f} GB'.format(x_valid.nbytes / 1024.**3 ))

		
		# continue training from a broken run, this loads last state of optimizer too!
		last_model = sorted(glob(f"/home/ubuntu/draw/weights/{self.exp_name}*.hdf5"), 
			key=os.path.getmtime,reverse=True)		

		if continue_training and len(last_model) > 0:
			# appending a token to indicate continued experiment
			self.exp_name = self.exp_name + "x"*len(last_model)

			last_model = last_model[0]
			self.model = keras.models.load_model(last_model,
												 custom_objects={"top_3_accuracy":self.top_3_accuracy})   
		
			print (f"Found existing model: {last_model}")
			print(f"Continuing training, new exp_name: {self.exp_name}")

		else:
			# important to put this under else condition otherwise learning rate will be reset
			self.model.compile(loss='categorical_crossentropy',
			  optimizer=Adam(lr=2e-3),
			  metrics=['accuracy', self.top_3_accuracy])   			

		# override lr if specified
		if lr:
			self.model.compile(loss='categorical_crossentropy',
			  optimizer=Adam(lr=lr),
			  metrics=['accuracy', self.top_3_accuracy]) 


		callbacks = self.get_callbacks(self.exp_name, self.batch_size)

		train_datagen = image_generator_1d(size=self.imsize, 
			batchsize=self.batch_size, ks=range(NCSVS - 1),preprocess_input=self.preprocess_input)

		self.model.fit_generator(train_datagen,
			 steps_per_epoch=4096,
			 epochs = 10000,
			 validation_data = (x_valid, y_valid),
			 callbacks = callbacks,
			 verbose = 1, 
			 workers=1, #self.num_workers, 
			 max_queue_size=100,
			 use_multiprocessing=False) 




class Experiment_Sample:
	def __init__(self, 
				 imsize, 
				 batch_size, 
				 exp_name,
				 data_generator=CNN2D_generator,
				 preprocess_input=None,
				 min_lr=1e-5,
				 num_workers = None 
				 ):
		
		self.imsize = imsize
		self.batch_size = batch_size
		self.exp_name = exp_name     
		self.data_generator = data_generator
		self.preprocess_input = preprocess_input
		self.min_lr = min_lr
		self.num_workers = num_workers
		self.num_workers = self.num_workers or os.cpu_count()//2 - 1

	def fit_gen(self, num_epochs=100):

		# self.parallel_model.__setattr__('callback_model', self.model)
		callbacks = self.get_callbacks(self.exp_name, self.batch_size)

		self.model.fit_generator(self.train_gen,
			 steps_per_epoch=len(self.train_gen),
			 epochs = num_epochs,
			 validation_data = self.valid_gen,
			 validation_steps=len(self.valid_gen),  
			 callbacks = callbacks,
			 verbose = 1, 
			 workers=self.num_workers, 
			 max_queue_size=50,
			 use_multiprocessing=True) 


	def train(self, 
			  model,
			  train_df,
			  valid_df,
			 continue_training=True,
			 stroke_aug_fn=None,
			 iaa_seq=None, 
			 do_aug=False, 
			 preprocess_input=None, 
			 lr=None, 
			 reduce_lr_mode="sample",
			 finetune=False):
		
		self.model = model
		self.preprocess_input = preprocess_input or self.preprocess_input

		self.train_gen = self.data_generator(
			train_df.drawing,
			train_df.target,
			self.batch_size, 
			is_train_mode = do_aug,
			imsize=self.imsize,
			stroke_aug_fn=stroke_aug_fn,
			iaa_seq=iaa_seq, 
			preprocess_input=self.preprocess_input)

		self.valid_gen = self.data_generator(
			valid_df.drawing,
			valid_df.target,
			self.batch_size,
			is_train_mode=False, 
			imsize=self.imsize,
			preprocess_input=self.preprocess_input) 
		
		# continue training from a broken run, this loads last state of optimizer too!
		last_model = sorted(glob(f"/home/ubuntu/draw/weights/{self.exp_name}*.hdf5"), 
			key=os.path.getmtime,reverse=True)		

		if continue_training and len(last_model) > 0:
			# appending a token to indicate continued experiment
			self.exp_name = self.exp_name + "x"*len(last_model)
						
			last_model = last_model[0]
			self.model = keras.models.load_model(last_model,
												 custom_objects={"top_3_accuracy":self.top_3_accuracy})   
		

			print(f"Continuing training, new exp_name: {self.exp_name}")

		else:
			# important to put this under else condition otherwise learning rate will be reset
			self.model.compile(loss='categorical_crossentropy',
			  optimizer=Adam(lr=2e-3),
			  metrics=['accuracy', self.top_3_accuracy])   			

		# override lr if specified
		if lr:
			self.model.compile(loss='categorical_crossentropy',
			  optimizer=Adam(lr=lr),
			  metrics=['accuracy', self.top_3_accuracy]) 


		if not finetune:
			self.fit_gen(num_epochs=100)

		else:
			# first train top layers
			self.fit_gen(num_epochs=5)

			# todo load weights 

			# unfreeze lower layers and retrain
			for layer in self.model.layers:
				layer.trainable = True				

			print(self.model.summary())
			print("Beginning tuning all layers .. ")

			self.model.compile(loss='categorical_crossentropy',
			  optimizer=Adam(lr=1e-3),
			  metrics=['accuracy', self.top_3_accuracy]) 

			self.fit_gen(num_epochs=100)

		

	def predict(self):
		last_model = sorted(glob(f"/home/ubuntu/draw/weights/{self.exp_name}*.hdf5"), 
							  key=os.path.getmtime,reverse=True)   
		if len(last_model) > 0:
			last_model=last_model[0]
		else:
			print("Prediction error: model doesn't exist for the given exp_name")
			return -1
		
		print (f"Predicting using model found in {last_model} ..")
		model = keras.models.load_model(last_model,
											 custom_objects={"top_3_accuracy":self.top_3_accuracy})        


		test_df = pd.read_csv('/home/ubuntu/draw/data/test_simplified.csv')
		test_gen = self.data_generator(test_df.drawing.values,
													 np.zeros(test_df.shape[0]),
													 self.batch_size, 
													 is_train_mode=False,
													 imsize=self.imsize, 
													 preprocess_input=self.preprocess_input)     
		
		test_predictions = model.predict_generator(test_gen, 
												   steps=len(test_gen), 
												   verbose=1)
		test_predictions = test_predictions[:test_df.shape[0]]
		np.save(f"/home/ubuntu/draw/predictions/{self.exp_name}.npy",test_predictions)
		

	def top_3_accuracy(self, x,y): 
		t3 = top_k_categorical_accuracy(x,y, 3)
		return t3

	def get_callbacks(self, exp_name, batch_size):
		reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, 
										   verbose=1, mode='auto', min_delta=0.01, min_lr=self.min_lr)
		# earlystop = EarlyStopping(monitor='val_top_3_accuracy', mode='max', patience=5) 
		earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=5) 

		exp_path = f'/home/ubuntu/draw/weights/{exp_name}.hdf5'
		save_best = ModelCheckpoint(monitor='val_loss',
								 filepath=exp_path, #+'-{val_loss:.2f}.hdf5',
								 save_best_only=True,
								 save_weights_only=False,
								 mode='min')
		csv_logger = CSVLogger('/home/ubuntu/draw/logs/csv/{}_train_log.csv'.format(exp_name))

		callbacks = [reduceLROnPlat, earlystop, save_best,csv_logger]
		return callbacks










		# ok now we have at least a running single gpu model, try to parallalize
		# try:
		# 	self.parallel_model = multi_gpu_model(self.model, cpu_relocation=True)
		# 	self.parallel_model.layers[-2].set_weights(self.model.get_weights())
		# 	print("Training using multiple GPUs..")

		# except ValueError:
		# 	self.parallel_model = self.model
		# 	print("Training using single GPU or CPU..")


# class MultiGPUCheckpoint(ModelCheckpoint):
	
#     def set_model(self, model):
#         if isinstance(model.layers[-2], Model):
#             self.model = model.layers[-2]
#         else:
#             self.model = model



		# tb_callback = TensorBoard(log_dir='/home/ubuntu/draw/logs/tensor_board/', 
		# 			histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False)


