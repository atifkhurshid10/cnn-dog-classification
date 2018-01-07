import numpy as np
import random
import scipy.misc
from PIL import Image

SEPERATOR = 750
TRAIN_DATA_SIZE = 4728
TEST_DATA_SIZE = 250
IMAGE_SIZE = 32
NUM_CLASSES = 25

def label_to_onehot(label):
	onehot = np.zeros(NUM_CLASSES)
	label = int(label)
	if label >= 0 and label <= NUM_CLASSES - 1:
		onehot[label] = 1
		return onehot
	else:
		print("Incorrect label!")

def resize(batchsize, original, shapes):
	resized = []
	for i in xrange(batchsize):
		x = np.reshape(original[i],(shapes[i][1], shapes[i][0], 3))
		resized.append(scipy.misc.imresize(x,(IMAGE_SIZE,IMAGE_SIZE), mode = "RGB"))
	return resized

def load_data():
	train_original = []
	train_shapes = []
	train_labels = []
	
	train_dictionary = np.load("../training.npy").item()
	for i in xrange(TRAIN_DATA_SIZE):
		train_original.append(train_dictionary["original"][i].astype(np.uint8))
		train_shapes.append(train_dictionary["shapes"][i])
		train_labels.append(label_to_onehot(train_dictionary["label"][i]))
	
	train_seeds = range(TRAIN_DATA_SIZE)
	random.shuffle(train_seeds)

	train_original = [train_original[i] for i in train_seeds]
	train_shapes = [train_shapes[i] for i in train_seeds]
	train_labels = [train_labels[i] for i in train_seeds]
	
	vali_original = train_original[:SEPERATOR]
	vali_shapes = train_shapes[:SEPERATOR]
	vali_labels = train_labels[:SEPERATOR]

	train_original_r = train_original[SEPERATOR:]
	train_shapes_r = train_shapes[SEPERATOR:]
	train_labels_r = train_labels[SEPERATOR:]


	return train_original, vali_original, train_shapes, vali_shapes, train_labels, vali_labels

def load_test_data():
	original = []
	shapes = []
	labels = []
	dictionary = np.load("../testing.npy").item()
	for i in xrange(TEST_DATA_SIZE):
		original.append(dictionary["original"][i].astype(np.uint8))
		shapes.append(dictionary["shapes"][i])
		labels.append(label_to_onehot(dictionary["labels"][i]))

	return original, shapes, labels

def batch(batchsize, o, s, l, mode='train'):
	original = []
	shapes = []
	labels = []
	if mode == 'train':
		seeds = range(TRAIN_DATA_SIZE - SEPERATOR)
	else if mode = 'vali':
		seeds = range(SEPERATOR)
	else:
		print "Preprocess.py: function batch(): Incorrect mode"
	random.shuffle(seeds)
	seeds = seeds[:batchsize]

	for i in seeds:
		original.append(o[i])
		shapes.append(s[i])
		labels.append(l[i])

	resized = resize(batchsize, original, shapes)
	return resized, labels

def test_batch(batchsize, original, shapes, labels):
	x = []
	s = []
	l = []
	for i in xrange(batchsize):
		x.append(original[i])
		s.append(shapes[i])
		l.append(labels[i])
	resized = resize(batchsize, x, s)
	return resized, l

