from utils import *
from qd_data import *

class Experiment_Sample:
	def __init__(self, 
				 imsize, 
				 batch_size, 
				 exp_name,
				 data_generator=QD_Datagen_Sample,
				 preprocess_input=keras.applications.mobilenet_v2.preprocess_input):
		
		self.imsize = imsize
		self.batch_size = batch_size
		self.exp_name = exp_name     
		self.data_generator = data_generator
		self.preprocess_input = preprocess_input

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
			 num_workers = None ):
		
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
			  optimizer=Adam(lr=2*1e-4),
			  metrics=['accuracy', self.top_3_accuracy])   			

		# override lr if specified
		if lr:
			self.model.compile(loss='categorical_crossentropy',
			  optimizer=Adam(lr=lr),
			  metrics=['accuracy', self.top_3_accuracy]) 


		# self.parallel_model.__setattr__('callback_model', self.model)
		callbacks = self.get_callbacks(self.exp_name, self.batch_size,reduce_lr_mode)
		num_workers = num_workers or os.cpu_count()//2 -1
		print ("num_workers:", num_workers)

		self.model.fit_generator(self.train_gen,
			 steps_per_epoch=len(self.train_gen)//(self.batch_size),
			 epochs = 100,
			 validation_data = self.valid_gen,
			 validation_steps=np.ceil(len(self.valid_gen)/(self.batch_size)), #10,  
			 callbacks = callbacks,
			 verbose = 1, 
			 workers=os.cpu_count()//2-1, 
			 max_queue_size=100,
			 use_multiprocessing=True) 
		

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
												   steps=np.ceil(len(test_df)/self.batch_size), 
												   verbose=1)
		test_predictions = test_predictions[:test_df.shape[0]]
		np.save(f"/home/ubuntu/draw/predictions/{self.exp_name}.npy",test_predictions)
		

	def top_3_accuracy(self, x,y): 
		t3 = top_k_categorical_accuracy(x,y, 3)
		return t3

	def get_callbacks(self, exp_name, batch_size):
		reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, 
										   verbose=1, mode='auto', min_delta=0.01, min_lr=1e-5)
		# earlystop = EarlyStopping(monitor='val_top_3_accuracy', mode='max', patience=5) 
		earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=5) 

		exp_path = f'/home/ubuntu/draw/weights/{exp_name}.hdf5'
		save_best = ModelCheckpoint(monitor='val_loss',
								 filepath=exp_path, #+'-{val_loss:.2f}.hdf5',
								 save_best_only=True,
								 save_weights_only=False,
								 mode='min')
		csv_logger = CSVLogger('/home/ubuntu/draw/logs/csv/{}_train_log.csv'.format(exp_name))
		# tb_callback = TensorBoard(log_dir='/home/ubuntu/draw/logs/tensor_board/', 
		# 			histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False)
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





# class Experiment:
# 	def __init__(self, 
# 				 imsize, 
# 				 batch_size, 
# 				 exp_name):
		
# 		self.imsize = imsize
# 		self.batch_size = batch_size
# 		self.exp_name = exp_name     
		

# 	def train(self, 
# 			  model,
# 			  train_df,
# 			  valid_df,
# 			 continue_training=True,
# 			 do_aug=True,
# 			 lr=None):
		
# 		self.model = model
# 		# self.model.compile(loss='categorical_crossentropy',
# 		#   optimizer='adam',
# 		#   metrics=['accuracy', self.top_3_accuracy])     			

# 		self.train_gen = QD_Datagen(train_df.drawing,
# 			train_df.target,
# 			self.batch_size, 
# 			is_train_mode = do_aug,
# 			imsize=self.imsize)
# 		self.valid_gen = QD_Datagen(valid_df.drawing,
# 			valid_df.target,
# 			self.batch_size,
# 			is_train_mode=False,
# 			imsize=self.imsize) 

# 		# continue training from a broken run, this loads last state of optimizer too!
# 		last_model = sorted(glob(f"/home/ubuntu/draw/weights/{self.exp_name}*.hdf5"), 
# 			key=os.path.getmtime,reverse=True)		

# 		if continue_training and len(last_model) > 0:
# 			# appending a token to indicate continued experiment
# 			self.exp_name = self.exp_name + "x"*len(last_model)
						
# 			last_model = last_model[0]
# 			self.model = keras.models.load_model(last_model,
# 												 custom_objects={"top_3_accuracy":self.top_3_accuracy})   
		

# 			print(f"Continuing training, new exp_name: {self.exp_name}")

# 		else:
# 			# important to put this under else condition otherwise learning rate will be reset
# 			self.model.compile(loss='categorical_crossentropy',
# 			  optimizer='adam',
# 			  metrics=['accuracy', self.top_3_accuracy])   			

# 		if lr:
# 			self.model.compile(loss='categorical_crossentropy',
# 			  optimizer=Adam(lr=lr),
# 			  metrics=['accuracy', self.top_3_accuracy])   	

# 		# ok now we have at least a running single gpu model, try to parallalize
# 		# try:
# 		# 	self.parallel_model = multi_gpu_model(self.model, cpu_relocation=True)
# 		# 	self.parallel_model.layers[-2].set_weights(self.model.get_weights())
# 		# 	print("Training using multiple GPUs..")

# 		# except ValueError:
# 		# 	self.parallel_model = self.model
# 		# 	print("Training using single GPU or CPU..")

# 		# self.model.__setattr__('callback_model', self.model)
# 		callbacks = self.get_callbacks(self.exp_name, self.batch_size)
		
# 		self.model.fit_generator(self.train_gen,
# 			 steps_per_epoch=len(self.train_gen)//(self.batch_size),
# 			 epochs = 100,
# 			 validation_data = self.valid_gen,
# 			 validation_steps=np.ceil(len(self.valid_gen)/(self.batch_size)), #10,  
# 			 callbacks = callbacks,
# 			 verbose = 1, 
# 			 workers=os.cpu_count()//2, 
# 			 max_queue_size=50,
# 			 use_multiprocessing=True) 
		
# 	def predict(self, num_augmentations=0):
# 		last_model = sorted(glob(f"/home/ubuntu/draw/weights/{self.exp_name}*.hdf5"), 
# 							  key=os.path.getmtime,reverse=True)   
# 		if len(last_model) > 0:
# 			last_model=last_model[0]
# 		else:
# 			print("Prediction error: model doesn't exist for the given exp_name")
# 			return -1
		
# 		print (f"Predicting using model found in {last_model} ..")
# 		model = keras.models.load_model(last_model,
# 											 custom_objects={"top_3_accuracy":self.top_3_accuracy})        


# 		test_df = pd.read_csv('/home/ubuntu/draw/data/test_simplified.csv')
# 		test_gen = QD_Datagen(test_df.drawing.values,
# 													 np.zeros(test_df.shape[0]),
# 													 self.batch_size, 
# 													 is_train_mode=False,
# 													 imsize=self.imsize)     
		
# 		test_predictions = model.predict_generator(test_gen, 
# 												   steps=np.ceil(len(test_df)/self.batch_size), 
# 												   verbose=1)
# 		test_predictions = test_predictions[:test_df.shape[0]]

# 		np.save(f"/home/ubuntu/draw/predictions/{self.exp_name}.npy",test_predictions)
		

# 		# avoid misuse bugs
# 		augmented_predictions = []
# 		augmented_predictions.append(test_predictions)
# 		del test_predictions

# 		if num_augmentations > 0:
			
# 			for n in range(num_augmentations):
# 				test_gen = QD_Datagen(test_df.drawing.values,
# 															 np.zeros(test_df.shape[0]),
# 															 self.batch_size, 
# 															 is_train_mode=True,
# 															 imsize=self.imsize)  

# 				augp = model.predict_generator(test_gen, 
# 														   steps=np.ceil(len(test_df)/self.batch_size), 
# 														   verbose=1)
# 				augp = augp[:test_df.shape[0]]
# 				augmented_predictions.append(augp)

# 			augmented_predictions = np.mean(augmented_predictions,axis=0)
# 			np.save(f"/home/ubuntu/draw/predictions/{self.exp_name}_augx{num_augmentations}.npy",augmented_predictions)


# 	def top_3_accuracy(self, x,y): 
# 		t3 = top_k_categorical_accuracy(x,y, 3)
# 		return t3

# 	def get_callbacks(self, exp_name, batch_size):
# 		reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, 
# 										   verbose=1, mode='auto', min_delta=0.01, min_lr=0.00001)
# 		# earlystop = EarlyStopping(monitor='val_top_3_accuracy', mode='max', patience=5) 
# 		earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=5) 

# 		exp_path = f'/home/ubuntu/draw/weights/{exp_name}.hdf5'
# 		save_best = ModelCheckpoint(monitor='val_loss',
# 								 filepath=exp_path, #+'-{val_loss:.2f}.hdf5',
# 								 save_best_only=True,
# 								 save_weights_only=False,
# 								 mode='min')
# 		csv_logger = CSVLogger('/home/ubuntu/draw/logs/csv/{}_train_log.csv'.format(exp_name))
# 		tb_callback = TensorBoard(log_dir='/home/ubuntu/draw/logs/tensor_board/', 
# 					histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False)
# 		callbacks = [reduceLROnPlat, earlystop, save_best,csv_logger,tb_callback]
# 		return callbacks

