import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label


# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy


Train_dir = 'D:/Hackerfest2018/ISIC2018_Task3_Training_Input/'
Test_dir = 'D:/Hackerfest2018/ISIC2018_Task3_Test_Input/'
Val_dir = 'D:/Hackerfest2018/ISIC2018_Task3_Validation_Input/'

Training_labels = pd.read_csv('D:/Hackerfest2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv')

train_ids = next(os.walk(Train_dir))[2]
test_ids = next(os.walk(Test_dir))[2]
val_ids = next(os.walk(Val_dir))[2]

'''
Label_IDs = Training_labels['image']
Melanoma = Training_labels['MEL']
NV = Training_labels['NV']
BCC = Training_labels['BCC']
AKIEC = Training_labels['AKIEC']
BKL = Training_labels['BKL']
DF = Training_labels['DF']
VASC = Training_labels['VASC']

Training_labels = Training_labels.drop(columns=['image'])

# Get and resize train images

X_train = np.zeros((46543, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((46543, 7), dtype=int)

print('Getting and resizing train images')

sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
	path = Train_dir + id_
	id_ = id_.replace('.jpg', '')
	img = imread(path)[:,:,:IMG_CHANNELS]
	img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
	for row, ID in enumerate(Label_IDs):
		if ID == id_:
			


#Get and resize val images
X_val = np.zeros((len(val_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
print('Getting and resizing val images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(val_ids), total=len(val_ids)):
	path = Val_dir + id_
	img = imread(path)[:,:,:IMG_CHANNELS]
	img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
	X_val[n] = img

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
#sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = Test_dir + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    #sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

#np.save('training_data.npy', X_train)
#np.save('training_labels.npy', Y_train)
#np.save('testing_data.npy', X_test)
#np.save('validation_data.npy', X_val)

#print('Done!')
'''
'''
X_train = np.load('D:/Hackerfest2018/Processed_Data/training_data.npy')
Y_train = np.load('D:/Hackerfest2018/Processed_Data/training_labels.npy')
X_test = np.load('D:/Hackerfest2018/Processed_Data/testing_data.npy')
X_val = np.load('D:/Hackerfest2018/Processed_Data/validation_data.npy')

Melanoma = 0
NV = 0
BCC = 0
AKIEC = 0
BKL = 0
DF = 0
VASC = 0

for list_ in Y_train:
	if list_[0] == 1:
		Melanoma += 1
	elif list_[1] == 1:
		NV += 1
	elif list_[2] == 1:
		BCC += 1
	elif list_[3] == 1:
		AKIEC += 1
	elif list_[4] == 1:
		BKL += 1
	elif list_[5] == 1:
		DF += 1	
	elif list_[6] == 1:
		VASC += 1

per_M = Melanoma / len(Y_train)					
per_NV = NV / len(Y_train)
per_BCC = BCC / len(Y_train)
per_AKIEC = AKIEC / len(Y_train)
per_BKL = BKL / len(Y_train)
per_DF = DF / len(Y_train)
per_VASC = VASC / len(Y_train)


print(Melanoma)
print(NV)
print(BCC)
print(AKIEC)
print(BKL)
print(DF)
print(VASC)


print(per_NV / per_M)
print(per_NV / per_BCC)
print(per_NV / per_AKIEC)
print(per_NV / per_BKL)
print(per_NV / per_DF)
print(per_NV / per_VASC)
'''
