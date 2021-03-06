from utils import *


class BaseGenerator(Sequence):
	"""docstring for BaseGenerator"""

	def __init__(self, 
		x_set, 
		y_set, 
		batch_size, 
		imsize, 
		is_train_mode=False, 
		stroke_aug_fn=None, 
		iaa_seq = None,
		preprocess_input=None
		):

		self.x, self.y = x_set, y_set
		self.batch_size = batch_size
		self.is_train_mode = is_train_mode
		self.imsize = imsize
		self.get_augmented_strokes = stroke_aug_fn
		self.iaa_seq = iaa_seq
		self.preprocess_input = preprocess_input

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

	def draw_it_3channel(self,strokes):

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
		image = np.array(image,dtype=np.float32)

  

		image = np.tile(image[:, :, None], [1, 1, 3])
		
		image = self.preprocess_input(image)
		return image


	def draw_it_3channel_augv3(self, strokes):
		image = Image.new("P", (300,300), color=255)
		image_draw = ImageDraw.Draw(image)

		# affine translation
	#     (300-256)//2
		pad= (300-256)//2
		offset_x = np.random.randint(5,2*pad-5)
		offset_y = np.random.randint(5,2*pad-5) #np.random.randint(150-pad +5,150+pad-5)

		# affine rotation
		rotation_val =  math.radians(np.random.uniform(-10,10))
		strokes = ast.literal_eval(strokes)

		for stroke_order,stroke in enumerate(strokes):

			# adding local jitter per stroke
			local_offset_x = np.random.randint(-2,2)
			local_offset_y = np.random.randint(-2,2)
			local_rotation = math.radians(np.random.uniform(-2,2))


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
		image = np.array(image,dtype=np.float32)   

		# horizontal flip
		if np.random.rand() < 0.5:
			image = np.flip(image,axis=1)    
		
		image = np.tile(image[:, :, None], [1, 1, 3])

		image = self.preprocess_input(image)
		return image


	def get_point_seqx3(self,raw_strokes):
		max_length = 196
		stroke_vec = literal_eval(raw_strokes) # string->list
		# unwrap the list
		in_strokes = [(xi,yi,i)  
		 for i,(x,y) in enumerate(stroke_vec) 
		 for xi,yi in zip(x,y)]
		c_strokes = np.stack(in_strokes)
		# replace stroke id with 1 for continue, 2 for new
		c_strokes[:,2] = [1]+np.diff(c_strokes[:,2]).tolist()
		c_strokes[:,2] += 1 # since 0 is no stroke
		# pad the strokes with zeros
		return pad_sequences(c_strokes.swapaxes(0, 1), 
							 maxlen=max_length, 
							 padding='post').swapaxes(0, 1)


	def get_stroke_sequence(self, strokes):
		strokes_concat = []
		count = 0 
		max_length = 20
	
		strokes = ast.literal_eval(strokes)

		for s_id in range(min(len(strokes),max_length)):
			stroke = strokes[s_id]
			image = Image.new("P", (256,256), color=255)
			image_draw = ImageDraw.Draw(image)        
			
			for i in range(len(stroke[0])-1):
				image_draw.line([stroke[0][i], 
								 stroke[1][i],
								 stroke[0][i+1], 
								 stroke[1][i+1]],
								fill=0, width=5)

			image = image.resize(( self.imsize,  self.imsize))
			image = np.array(image)/255.
			strokes_concat.append(image)
			
			count += 1 

		
		strokes_concat = np.hstack(strokes_concat)
		
		if count < max_length:
			pad_length = max_length - count
			pad = np.ones((self.imsize,pad_length*self.imsize))
			strokes_concat = np.hstack([strokes_concat,pad])

		strokes_concat = np.reshape(strokes_concat, strokes_concat.shape+(1,))	
		return strokes_concat 

		
	def get_discrete_stroke_sequence(self, strokes):
		strokes_concat = []
		count = 0 
		max_length = 20
	
		strokes = ast.literal_eval(strokes)

		for s_id in range(min(len(strokes),max_length)):
			stroke = strokes[s_id]
			image = Image.new("P", (256,256), color=255)
			image_draw = ImageDraw.Draw(image)        
			
			for i in range(len(stroke[0])-1):
				image_draw.line([stroke[0][i], 
								 stroke[1][i],
								 stroke[0][i+1], 
								 stroke[1][i+1]],
								fill=0, width=5)

			image = image.resize(( self.imsize,  self.imsize))
			image = np.array(image)/255.
			image = np.reshape(image, ( self.imsize,  self.imsize, 1))
			
			strokes_concat.append(image)
			
			count += 1 

		
		# strokes_concat = np.hstack(strokes_concat)
		
		for i in range((max_length-count)):
			filler = np.ones((self.imsize, self.imsize , 1))
			strokes_concat.append(filler)


		strokes_concat = np.array(strokes_concat)		
		return strokes_concat 

	def draw_it_3channel_infoplus(self,strokes):

		if self.is_train_mode and self.get_augmented_strokes:
			strokes = self.get_augmented_strokes(strokes)
		
		# R
		image = Image.new("P", (256,256), color=255)
		image_draw = ImageDraw.Draw(image)

		# G
		stroke_order_image = Image.new("P", (256,256), color=255)
		stroke_order_image_draw = ImageDraw.Draw(stroke_order_image)		

		# B
		point_order_image = Image.new("P", (256,256), color=255)
		point_order_image_draw = ImageDraw.Draw(point_order_image)		

		point_count = 0
		for stroke_id, stroke in enumerate(ast.literal_eval(strokes)):
			for i in range(len(stroke[0])-1):
				x1 = stroke[0][i]
				y1 = stroke[1][i]
				x2 = stroke[0][i+1]
				y2 = stroke[1][i+1]

				image_draw.line([x1,y1,x2,y2],
								fill=0, width=5)

				point_count += 1

				point_order_image_draw.line([x1,y1,x2,y2],  fill=min((point_count),255), width=5)
				stroke_order_image_draw.line([x1,y1,x2,y2], fill=min((stroke_id*5),255), width=5)
		
		
		image = image.resize(( self.imsize,  self.imsize))
		image = np.array(image,dtype=np.float32)
		
		stroke_order_image = stroke_order_image.resize(( self.imsize,  self.imsize))
		stroke_order_image = np.array(stroke_order_image,dtype=np.float32)
		
		point_order_image = point_order_image.resize(( self.imsize,  self.imsize))
		point_order_image = np.array(point_order_image, dtype=np.float32)

		# # horizontal flip
		# if np.random.rand() < 0.5:
		# 	image = np.flip(image,axis=1)  
		# 	stroke_order_image = np.flip(stroke_order_image,axis=1)
		# 	point_order_image = np.flip(point_order_image,axis=1)  
			
		image = np.stack([image, stroke_order_image, point_order_image],axis=-1)

		image = self.preprocess_input(image)
		return image



class RNN_1d_generator(BaseGenerator):

	def __getitem__(self, idx):
		batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
		batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

		# images = np.array([self.draw_it(strokes) for strokes in batch_x])
		points = np.array([self.get_point_seqx3(strokes) for strokes in batch_x])
		# strokes = np.array([self.get_point_sequence(strokes) for strokes in batch_x])

		return points, keras.utils.to_categorical(batch_y, num_classes)



class CRNN_2d_generator(BaseGenerator):

	def __getitem__(self, idx):
		batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
		batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

		images = np.array([self.draw_it_3channel_infoplus(strokes) for strokes in batch_x])
		points = np.array([self.get_point_seqx3(strokes) for strokes in batch_x])
		# strokes = np.array([self.get_discrete_stroke_sequence(strokes) for strokes in batch_x])

		return [points,images], keras.utils.to_categorical(batch_y, num_classes)


class CNN2D_generator(BaseGenerator):
	
	def __getitem__(self, idx):
		batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
		batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
		images = np.array([self.draw_it_3channel_infoplus(strokes) for strokes in batch_x])

		return images, keras.utils.to_categorical(batch_y, num_classes)



class TestGenerator(Sequence):

	def __init__(self, 
		df, 
		batch_size, 
		imsize, 
		preprocess_input=None
		):

		self.df = df
		self.batch_size = batch_size
		self.imsize = imsize
		self.preprocess_input = preprocess_input

	def __len__(self):
		return int(np.ceil(len(self.df) / float(self.batch_size)))	


	def __getitem__(self, idx):
		batch_x = self.df.iloc[idx * self.batch_size:(idx + 1) * self.batch_size].copy()
		return df_to_image_array_xd(batch_x, self.imsize, preprocess_input=self.preprocess_input)
