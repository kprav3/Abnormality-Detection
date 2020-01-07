import os
import numpy as np
import pandas as pd 
import random
import cv2
import matplotlib.pyplot as plt


import keras.backend as k
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf


class VGG1:
    @staticmethod
    def build(height, width, depth, classes):
        model = Sequential()
        input_shape = (height, width, depth)
        channel_dim = -1
        if (k.image_data_format() == 'channels_first'):
            input_shape = (depth, height, width)
            channel_dim = 1
            
        inputs = Input(shape=input_shape)

#input1= keras.layers.Input(shape=(96,96,3))
        x = Conv2D(filters=16, kernel_size=(3, 3), padding='same',input_shape=input_shape)(inputs)
        x = Activation("relu")(x)
        x = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(x)
        x = Activation("relu")(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

# Second conv block
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
        x = Activation("relu")(x)
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

# Third conv block
        x = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
        x = Activation("relu")(x)
        x = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)

# Fourth conv block
        x = SeparableConv2D(filters=128, kernel_size=(3, 3), padding='same')(x)
        x = Activation("relu")(x)
        x = SeparableConv2D(filters=128, kernel_size=(3, 3), padding='same')(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Dropout(rate=0.2)(x)

# Fifth conv block
        x = SeparableConv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
        x = Activation("relu")(x)
        x = SeparableConv2D(filters=256, kernel_size=(3, 3), padding='same')(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Dropout(rate=0.2)(x)

# FC layer
        x = Flatten()(x)
        x = Dense(units=512)(x)
        x = Activation("relu")(x)
        x = Dropout(rate=0.7)(x)
        x = Dense(units=128)(x)
        x = Activation("relu")(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(units=64)(x)
        x = Activation("relu")(x)
        x = Dropout(rate=0.3)(x)

# Output layer
        output = Dense(units=1)(x)
        x = Activation("sigmoid")(x)

        keras.layers.Concatenate(axis=-1)
# Creating model and compiling
        model = Model(inputs=inputs, outputs=output)


        return model 
