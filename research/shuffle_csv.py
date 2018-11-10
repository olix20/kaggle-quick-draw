from IPython.core.interactiveshell import InteractiveShell
import ast
import os
import datetime as dt
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys


def f2cat(filename: str) -> str:
	return filename.split('.')[0]

class Simplified():
	def __init__(self, input_path='/home/ubuntu/draw/data'):
		self.input_path = input_path

	def list_all_categories(self):
		files = os.listdir(os.path.join(self.input_path, 'train_simplified'))
		return sorted([f2cat(f) for f in files], key=str.lower)

	def read_training_csv(self, category, nrows=None, usecols=['key_id','drawing','word'], drawing_transform=False):
		df = pd.read_csv(os.path.join(self.input_path, 'train_simplified', category + '.csv'),
						 nrows=nrows, usecols=usecols) #parse_dates=['timestamp']
		if drawing_transform:
			df['drawing'] = df['drawing'].apply(ast.literal_eval)
		return df


start = dt.datetime.now()
s = Simplified()
NCSVS = 100
categories = s.list_all_categories()
print(len(categories))


# print ("creating CSVs .. ")
# for y, cat in tqdm(enumerate(categories)):
# 	df = s.read_training_csv(cat) #, nrows=30000
# 	df['y'] = y
# 	df['cv'] = (df.key_id // 10 ** 7) % NCSVS
# 	for k in range(NCSVS):
# 		filename = '../data/csv_gz/train_k{}.csv'.format(k)
# 		chunk = df[df.cv == k]
# 		chunk = chunk.drop(['key_id'], axis=1)
# 		if y == 0:
# 			chunk.to_csv(filename, index=False)
# 		else:
# 			chunk.to_csv(filename, mode='a', header=False, index=False)



offset = int(sys.argv[1])

print ("Shuffling and archiving CSVs .. ", offset)

for k in tqdm(range(offset,offset+10)):
	filename = '../data/csv_gz/train_k{}.csv'.format(k)
	if os.path.exists(filename):
		df = pd.read_csv(filename)
		df['rnd'] = np.random.rand(len(df))
		df = df.sort_values(by='rnd').drop('rnd', axis=1)
		df.to_csv(filename + '.gz', compression='gzip', index=False)
		os.remove(filename)
	else:
		print ("file doesn't exist")



print(df.shape)
