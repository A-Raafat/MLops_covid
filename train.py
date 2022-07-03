import numpy as np
import os
import sys
import cv2 
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import layers
import matplotlib.pyplot as plt

train_data_path='Data/train/'
test_data_path ='Data/test/'


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        train_data_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        test_data_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')



model = Sequential()

model.add(layers.Conv2D(filters=8, kernel_size=(5, 5), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Flatten())
#model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(units=120, activation='relu'))
model.add(layers.Dense(units=84, activation='relu'))
model.add(layers.Dense(units=1, activation = 'sigmoid'))


model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])


history = model.fit(
        train_generator,
        steps_per_epoch=56,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=15)


model.save("Covid_model")


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Accuracy_graph.jpg')



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Loss_graph.jpg')




