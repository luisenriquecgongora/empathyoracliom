import csv
import cv2
import numpy as np
from keras import backend as K
from keras.layers import Dense, Dropout,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Activation
from keras.utils import np_utils

dataf  = 0
lastime = '0'

X_train = []
Y_train = []
with open('dataunderstanding.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    i = 0
    for row in reader:
        dataf = row['pic']
        lastime = row['time']
        X_train
        imageData = np.load("pics/" + str(row['time']) + '.npy')
        X_train.append(imageData)
        understanding = int(row['understanding'])
        Y_train.append(understanding)
        i= i + 1
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_train = X_train.reshape(X_train.shape[0], 64, 64 , 1).astype('float32')
Y_train = Y_train.reshape(Y_train.shape[0], 1).astype('float32')
print(X_train.shape)
print(Y_train.shape)

X_train = (X_train - 128.0) / 255.0

y_train = np_utils.to_categorical(Y_train)

model = Sequential()

## CONVOLUTIONAL NEURAL NETWORK
model.add(Conv2D(40, kernel_size=5, padding="same",input_shape=(64, 64, 1), activation = 'relu'))
model.add(Conv2D(50, kernel_size=5, padding="valid", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(70, kernel_size=3, padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(100, kernel_size=3, padding="valid", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


model.add(Flatten())
model.add(Dense(100))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(2))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs= 16 , batch_size=40, validation_split = 0.2)
scores = model.evaluate(X_train, y_train, verbose = 10 )
print ( scores )

model.save('empathy_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model


#print(data)
## Define Initial Picture
### Get Pictures
### Get face data per picture
### Insert Face Data to MultiDStack
### Repeat
## Get Last Neurons Properties
## Train Each Neuron
## Define best ones
## Generate Sons
## Save Best Neurons
