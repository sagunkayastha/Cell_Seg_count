# -*- coding: utf-8 -*-


import numpy as np
import pdb
import os
import matplotlib.pyplot as plt
from generator import ImageDataGenerator
from seg_data import read_train_data
from seg_model import get_unet, dice_coef, dice_coef_loss
from keras import backend as K
from keras.callbacks import ModelCheckpoint,Callback,LearningRateScheduler
from scipy import misc
import scipy.ndimage as ndimage
from keras.models import model_from_json
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
import time as time 
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split

model_name = 'Segmentation'


def step_decay(epoch):
    step = 16
    num =  epoch // step 
    if num % 3 == 0:
        lrate = 1e-4
    elif num % 3 == 1:
        lrate = 1e-5
    else:
        lrate = 1e-6
        #lrate = initial_lrate * 1/(1 + decay * (epoch - num * step))
    print('Learning rate for epoch {} is {}.'.format(epoch+1, lrate))
    return np.float(lrate)


def train_():
    X, y = read_train_data()
    

    train_img, test_img, train_mask, test_mask     = train_test_split(X, y, test_size=0.2, random_state=1)

    train_img, val_img, train_mask, val_mask     = train_test_split(train_img, train_mask, test_size=0.2, random_state=1)
    
    print(train_img.shape, train_mask.shape)
    print('-'*30)
    print('UNET FOR MASK SEGMENTATION.')
    print('-'*30)    
   
    model = get_unet(IMG_WIDTH=256,IMG_HEIGHT=256,IMG_CHANNELS=1)
      
    model_checkpoint = ModelCheckpoint(model_name+".hdf5", monitor='loss', save_best_only=False)
    model.summary()


    model_json = model.to_json()
    with open("{}.json".format(model_name), "w") as json_file:
        json_file.write(model_json)

    print('...Fitting model...')
    print('-'*30)
    change_lr = LearningRateScheduler(step_decay)
    
    tensorboard = TensorBoard(log_dir="logs/{}".format(model_name))
    datagen = ImageDataGenerator(
        featurewise_center = False,  # set input mean to 0 over the dataset
        samplewise_center = False,  # set each sample mean to 0
        featurewise_std_normalization = False,  # divide inputs by std of the dataset
        samplewise_std_normalization = False,  # divide each input by its std
        zca_whitening = False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range = 0.3,  # randomly shift images horizontally (fraction of total width)
        height_shift_range = 0.3,  # randomly shift images vertically (fraction of total height)
        zoom_range = 0.3,
        shear_range = 0.,
        horizontal_flip = True,  # randomly flip images
        vertical_flip = True, # randomly flip images
        fill_mode = 'constant',
        dim_ordering = 'tf')  

    model.compile(optimizer='adam',loss='binary_crossentropy', metrics=[dice_coef])
    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(train_img,
                                     train_mask, 
                                     batch_size = 8
                                     ),
                        validation_data = (val_img,val_mask),
                        samples_per_epoch = train_img.shape[0],
                        nb_epoch = 10,                        
                        callbacks = [model_checkpoint, change_lr, tensorboard],
                       )
    
    score = model.evaluate(test_img, test_mask, batch_size=16)
    model.save_weights('seg_weight.h5')

def predict_():
    model = get_unet(IMG_WIDTH=256,IMG_HEIGHT=256,IMG_CHANNELS=1)
    try :
        model.load_weights('seg_weight.h5')
        X, y = read_train_data()
        train_img, test_img, train_mask, test_mask  = train_test_split(X, y, test_size=0.2, random_state=1)
        A = model.predict(test_img)
        np.save('A_.npy', A)
    except:
        pass
  
   

if __name__ == '__main__':
    train_()
    predict_()