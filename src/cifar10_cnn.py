from __future__ import print_function, absolute_import
from keras.datasets import cifar10
from image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from renet import *
from cifar import load_batch
from keras.utils.data_utils import get_file
import numpy as np
import os

batch_size = 32
nb_classes = 10
nb_epoch = 200
data_augmentation = True

img_rows, img_cols = 32, 32
img_channels = 3

path = "../data/cifar-10-batches-py"

nb_train_samples = 50000

X_train = np.zeros((nb_train_samples, 3, 32, 32), dtype="uint8")
y_train = np.zeros((nb_train_samples,), dtype="uint8")

for i in range(1, 6):
    fpath = os.path.join(path, 'data_batch_' + str(i))
    data, labels = load_batch(fpath)
    X_train[(i-1)*10000:i*10000, :, :, :] = data
    y_train[(i-1)*10000:i*10000] = labels

fpath = os.path.join(path, 'test_batch')
X_test, y_test = load_batch(fpath)

y_train = np.reshape(y_train, (len(y_train), 1))
y_test = np.reshape(y_test, (len(y_test), 1))

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


# this will do preprocessing and realtime data augmentation
datagen = ImageDataGenerator(
    featurewise_center=True,  # set input mean to 0 over the dataset
    featurewise_std_normalization=True,  # divide inputs by std of the dataset
    zca_whitening=True,  # apply ZCA whitening
    width_shift_range=0.25,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.25,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)  # randomly flip images
