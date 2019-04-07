import os
import random
import sys
import warnings
import numpy as np
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from keras.utils import Progbar
import matplotlib.pyplot as plt
import cv2
'''
SIMCEPImages_well_Ccells_Fblur;_ssamples_wstain.TIF
SIMCEPImages_A01_C1_F1_s01_w1

well -The standard 384-well plate format is used where the rows are named A-P and the columns 1- 24.
cells - The number of cells simulated in the image (1-100).
blur - The amount of focus blur applied (1-48). The focus blur was simulated by using MATLAB's imfilter function with a rotationally symmetric Gaussian lowpass filter of diameter <#2> and sigma of 0.25 × <#2>
sample - Number of samples (1-25) for a given combination of <#1> and <#2>. Can be used to mimic the "site" number for each well.
stain - 1 = cell body stain, 2 = nuclei stain.
'''


warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

# Setting seed for reproducability
seed = 42
random.seed = seed
np.random.seed = seed
files_up = False
if files_up == False:
    # Data Path
    # TRAIN_PATH = '../../BB_Data/BBBC005_v1_images/'
    LABEL_PATH = '../../BB_Data/BBBC005_v1_ground_truth/'

    # train_ids = next(os.walk(TRAIN_PATH))[2]
    label_ids = next(os.walk(LABEL_PATH))[2]



    ''' Only for cell Stain'''
    w1 = True
    if w1:
        label_ids = [i for i in label_ids if 'w1' in i]
    else:
        pass
    print(len(label_ids))


# Function read train images and mask return as nump array
def read_train_data(IMG_WIDTH=256,IMG_HEIGHT=256,IMG_CHANNELS=1):
    
    X_train = np.zeros((len(label_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    Y_train = np.zeros((len(label_ids)))
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    if os.path.isfile("train_img.npy") and os.path.isfile("train_mask.npy"):
        print("Train file loaded from memory")
        X_train = np.load("train_mask.npy")
        Y_train = np.load("train_count.npy")
        # print(X_train.shape, Y_train.shape)
        return X_train,Y_train
        
    print('Numpy file for Train Mask')
    a = Progbar(len(label_ids))
    for n, id_ in enumerate(label_ids):
        if 'w1' in id_:
            path = LABEL_PATH 
            img = imread(path  + id_ )
            nu_c=id_.split('_C')[1].split('_F')[0]   
                        
            np.expand_dims(resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                        preserve_range=True), axis=-1)
            
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True)
            X_train[n] = img
            Y_train[n] =nu_c
        a.update(n)
    
    print('-30', len(X_train))
         
    print('@#@',X_train.shape, Y_train.shape)
    
    np.save("train_mask",X_train)
    np.save("train_count",Y_train)
   
    
    return X_train,Y_train

# # Function to read test images and return as numpy array
# def read_test_data(IMG_WIDTH=256,IMG_HEIGHT=256,IMG_CHANNELS=3):
#     X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
#     sizes_test = []
#     print('\nGetting and resizing test images ... ')
#     sys.stdout.flush()
#     if os.path.isfile("test_img.npy") and os.path.isfile("test_size.npy"):
#         print("Test file loaded from memory")
#         X_test = np.load("test_img.npy")
#         sizes_test = np.load("test_size.npy")

#     for n, id_ in enumerate(test_ids):
#         path = TEST_PATH + id_
#         img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
#         sizes_test.append([img.shape[0], img.shape[1]])
#         img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
#         X_test[n] = img
#         b.update(n)
#     np.save("test_img",X_test)
#     np.save("test_size",sizes_test)
#     return X_test,sizes_test


if __name__ == '__main__':
    x,y = read_train_data()
    pass

