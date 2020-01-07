# 70 - 80

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras import backend as k

class CNN:
    @staticmethod
    def build(height, width, depth, classes):
    	classifier = Sequential()
    	input_shape = (height, width, depth)
    	channel_dim = -1
    	if (k.image_data_format() == 'channels_first'):
        	input_shape = (depth, height, width)
        	channel_dim = 1

    	classifier.add(Conv2D(32, (3, 3), input_shape = (28, 28, 3), activation = 'relu'))
    	classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(BatchNormalization(axis=channel_dim))
        classifier.add(MaxPooling2D(pool_size = (3,3)))
        classifier.add(Dropout(0.25))
        
        classifier.add(Conv2D(64,(3,3),padding='same'))
        classifier.add(Activation('relu'))
        classifier.add(BatchNormalization(axis=channel_dim))
        classifier.add(Conv2D(64,(3,3),padding='same'))
        classifier.add(Activation('relu'))
        classifier.add(BatchNormalization(axis=channel_dim))
    	classifier.add(Flatten())
    	classifier.add(Dense(units = 128, activation = 'relu'))
    	classifier.add(Dense(units = 1, activation = 'sigmoid'))
    	return classifier        
        
