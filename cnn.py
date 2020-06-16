 rnn# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:27:14 2020

@author: Ishparsh
"""

#importing libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#Initializing the CNN model
classifier=Sequential()

#Step 1 Convolution
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))

#Step2 Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#adding more layer
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))


#Step 3 Flattening
classifier.add(Flatten())

#Step 4 Full Connection
classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(128,activation='relu'))
classifier.add(Dense(1,activation='sigmoid'))

#compiling CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


#fitting Images into CNN
from keras.preprocessing.image import ImageDataGenerator
#before fitting we do image augmentation

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')


classifier.fit(
    training_set,
    steps_per_epoch=8000,
    epochs=2,
    validation_data=test_set,
    validation_steps=2000)

#checking the prediction
import numpy as np
from keras.preprocessing import image

test_image=image.load_img('dataset/predict/cat_or_dog.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)
training_set.class_indices