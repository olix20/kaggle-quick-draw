from utils import *



data_path ="/home/ubuntu/draw/data/"
train_path = data_path+"train/"
num_classes = 340 

class QD_Datagen(Sequence):
	def __init__(self, x_set, y_set, batch_size, imsize, is_train_mode=False):
		self.x, self.y = x_set, y_set
		self.batch_size = batch_size
		self.is_train_mode = is_train_mode
		self.imsize = imsize

	def __len__(self):
		return int(np.ceil(len(self.x) / float(self.batch_size)))
	
	def get_augmented_strokes(self, strokes, prob=0.1):
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

	def draw_it(self,strokes):

		# if self.is_train_mode:
		# 	strokes = self.get_augmented_strokes(strokes)
		
		image = Image.new("P", (256,256), color=255)
		image_draw = ImageDraw.Draw(image)
		for stroke in ast.literal_eval(strokes):
			for i in range(len(stroke[0])-1):
				image_draw.line([stroke[0][i], 
								 stroke[1][i],
								 stroke[0][i+1], 
								 stroke[1][i+1]],
								fill=0, width=5)
				
		image = image.resize(( self.imsize,  self.imsize))
		image = np.array(image)/255.
		
		return np.reshape(image,image.shape+(1,))
	
	
	def get_full_image_array(self):
		imagebag = bag.from_sequence(
			self.x,
			npartitions=os.cpu_count()-1).map(lambda x: self.draw_it(x))      
		raw_images = np.array(imagebag.compute())
		
		return raw_images
#         return np.array([
#             self.draw_it(strokes) for strokes in self.x])
		
	def __getitem__(self, idx):
		batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
		batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
		
		images = np.array([
			self.draw_it(strokes) for strokes in batch_x])
		
		# do augmentation
		if self.is_train_mode:
			# # sometimes = lambda aug: iaa.Sometimes(0.5, aug)
			# seq = iaa.Sequential([
			# 	iaa.Fliplr(0.5) # horizontally flip 50% of the images    
			# # 	sometimes(iaa.Affine(
			# # 		translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -10 to +10 percent (per axis)
			# # #         rotate=(-10, 10), # rotate by -45 to +45 degrees
			# # #         shear=(-1, 1), # shear by -16 to +16 degrees
			# # 		cval=1
			# # 	))   
			# 	])
			sometimes = lambda aug: iaa.Sometimes(0.5, aug)
			seq = iaa.Sequential([
				iaa.Fliplr(0.5),
				iaa.Flipud(0.2),
				sometimes(iaa.Affine(scale={"y": (0.8, 1.2)} ,  cval=1)),
				sometimes(iaa.Affine(translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},cval=1)),
				sometimes(iaa.Affine(shear=(-5, 5),  cval=1)),
				sometimes(iaa.Affine(rotate=(-10, 10),  cval=1))
				])			
			images = seq.augment_images(images)

		return images, keras.utils.to_categorical(batch_y, num_classes)
	








class QD_Datagen_Sample(Sequence):
	def __init__(self, 
		x_set, 
		y_set, 
		batch_size, 
		imsize, 
		is_train_mode=False, 
		stroke_aug_fn=None, 
		iaa_seq = None
		):

		self.x, self.y = x_set, y_set
		self.batch_size = batch_size
		self.is_train_mode = is_train_mode
		self.imsize = imsize
		self.get_augmented_strokes = stroke_aug_fn
		self.iaa_seq = iaa_seq

	def __len__(self):
		return int(np.ceil(len(self.x) / float(self.batch_size)))



	def draw_it_augv3(self, strokes, prob=0.1):

		if self.get_augmented_strokes:
			strokes = self.get_augmented_strokes(strokes)

		image = Image.new("P", (300,300), color=255)
		image_draw = ImageDraw.Draw(image)
		
		# affine translation
		offset_x = (300-256)//2 #np.random.randint(5,((300-256)//2)-5)
		offset_y = (300-256)//2 #np.random.randint(5,((300-256)//2)-5)
		
		# affine rotation
		rotation_val =  0 #math.radians(np.random.uniform(-10,10))
		strokes = ast.literal_eval(strokes)
		
		for stroke_order,stroke in enumerate(strokes):
			
			# adding local jitter per stroke
			local_offset_x = 0 #np.random.randint(-2,2)
			local_offset_y = 0 #np.random.randint(-2,2)
			local_rotation =  0 #math.radians(np.random.uniform(-2,2))
			
			
			for i in range(len(stroke[0])-1):
				x1 = stroke[0][i]+offset_x+local_offset_x
				y1 = stroke[1][i]+offset_y+local_offset_y
				x2 = stroke[0][i+1]+offset_x+local_offset_x
				y2 = stroke[1][i+1]+offset_y+local_offset_y
				
				x1, y1 = rotate_point((150,150),(x1,y1),rotation_val+local_rotation)
				x2, y2 = rotate_point((150,150),(x2,y2),rotation_val+local_rotation)
				image_draw.line([x1, 
								 y1,
								 x2, 
								 y2],
								fill=0, width=5)
				
		image = image.resize(( self.imsize,  self.imsize))
		image = np.array(image)/255.
		
		# horizontal flip
		if np.random.rand() < 0.5:
			image = np.flip(image,axis=1)
			
		return np.reshape(image,image.shape+(1,))


	# def draw_it(self, strokes):

	# 	image = Image.new("P", (300,300), color=255)
	# 	image_draw = ImageDraw.Draw(image)

	# 	offset = (300-256)//2

	# 	for stroke in ast.literal_eval(strokes):
	# 		for i in range(len(stroke[0])-1):
	# 			image_draw.line([stroke[0][i]+offset, 
	# 							 stroke[1][i]+offset,
	# 							 stroke[0][i+1]+offset, 
	# 							 stroke[1][i+1]+offset],
	# 							fill=0, width=5)
				
	# 	image = image.resize(( self.imsize,  self.imsize))
	# 	image = np.array(image)/255.
				
	# 	return np.reshape(image,image.shape+(1,))
	
	def draw_it(self,strokes):

		if self.is_train_mode and self.get_augmented_strokes:
			strokes = self.get_augmented_strokes(strokes)
		
		image = Image.new("P", (256,256), color=255)
		image_draw = ImageDraw.Draw(image)
		for stroke in ast.literal_eval(strokes):
			for i in range(len(stroke[0])-1):
				image_draw.line([stroke[0][i], 
								 stroke[1][i],
								 stroke[0][i+1], 
								 stroke[1][i+1]],
								fill=0, width=5)
				
		image = image.resize(( self.imsize,  self.imsize))
		image = np.array(image)/255.
		
		return np.reshape(image,image.shape+(1,))	
	
	def __getitem__(self, idx):
		batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
		batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
		
		# if self.is_train_mode:
		# 	images = np.array([
		# 				self.draw_it_augv3(strokes) for strokes in batch_x])
		# else:
		# 	images = np.array([
		# 				self.draw_it(strokes) for strokes in batch_x])
		images = np.array([self.draw_it(strokes) for strokes in batch_x])
		# # do augmentation
		if self.is_train_mode and self.iaa_seq:			
			images = self.iaa_seq.augment_images(images)

		return images, keras.utils.to_categorical(batch_y, num_classes)
	
