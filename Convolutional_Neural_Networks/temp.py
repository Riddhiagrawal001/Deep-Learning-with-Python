# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#part-1 Building the cnn
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#inilializing a cnn
classifier=Sequential()

#step1-Convolution
classifier.add(Conv2D(32 ,(3 , 3) , input_shape = (64,64,3) ,activation='relu'))

#step-2 pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#adding another convolutional layer so as to increase the test set accuracy
classifier.add(Conv2D(32 ,(3 , 3) ,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#step-3 Flattening
classifier.add(Flatten())

#step-4 Full connection
classifier.add(Dense(output_dim=128,activation = 'relu'))
classifier.add(Dense(output_dim=1,activation = 'sigmoid'))

#compiling the cnn
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#part-2 fitting the cnn to images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64,64),
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

#part-3 making new predictions
import numpy as np
from keras.preprocessing import image

test_image=image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size=(64, 64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)

if result[0][0]==1:
    prediction='dog'
else:
    prediction='cat'












































