import tensorflow as tf
from keras.utils import np_utils
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,\
  MaxPooling2D, BatchNormalization
from keras.layers.convolutional import Conv2D
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint,Callback


from count_data import read_train_data

X, y = read_train_data()
X=X[:600]

# x_train, x_test_pre, y_train, y_test_pre = train_test_split(X, Y, test_size=0.20, random_state=42)
# x_test, x_validation, y_test, y_validation = train_test_split(x_test_pre, y_test_pre, test_size=0.1)
from keras.layers import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint,Callback,LearningRateScheduler

image_gen = ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    
    rotation_range=15,
    width_shift_range=.15,
    height_shift_range=.15,
    horizontal_flip=True
)
image_gen.fit(X)

def step_decay(epoch):
    if epoch < 200:
        lrate = 1e-2
    elif epoch < 500:
        lrate = 1e-3
    elif epoch < 700:
        lrate = 1e-4
    else:
        lrate = 1e-5
        #lrate = initial_lrate * 1/(1 + decay * (epoch - num * step))
    print('Learning rate for epoch {} is {}.'.format(epoch+1, lrate))
    return np.float(lrate)

  
change_lr = LearningRateScheduler(step_decay)  

model_name = "count_weight"


model = Sequential()

#Conv 1
model.add(Conv2D(filters=64, input_shape=(256,256,1), kernel_size=(3,3),\
 strides=(1,1), padding='same',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))


# Pooling 
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))


#Conv 2
model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))


# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))


#Conv 3
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))


# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#Conv4
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))


# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

#Conv5
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))
model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))


# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

model.add(Flatten())
model.add(Dense(1024,input_shape=(8*8*512,),kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))


model.add(Dense(512,kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))

model.add(Dense(1,activation='relu'))


model.summary()

model_json = model.to_json()
with open("{}.json".format(model_name), "w") as json_file:
    json_file.write(model_json)


try:
    model.load_weights('count_weight.h5')
    print('Model LOafed')
except:
    pass

model_checkpoint = ModelCheckpoint(model_name+".h5", monitor='loss', save_best_only=False)


batch_size = 4
nb_epoch = 30
learn_rate = 0.01

adm = keras.optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=adm,loss='mse',metrics=['accuracy'])
model.fit_generator(image_gen.flow(X,y,batch_size = batch_size
                                        ),
                            
                            samples_per_epoch = X.shape[0],
                            nb_epoch =  nb_epoch,
                   callbacks = [model_checkpoint])

  

model.save_weights(model_name+'.h5')

out = model.predict(X)
for i in range(len(out)):
    print(f'out - {out[i]}, Label - {y[i]}')
