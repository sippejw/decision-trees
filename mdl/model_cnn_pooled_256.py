### model subsample pooling

print("importing requirements")
#basic imports
import numpy as np
import pandas as pd
import os
import sys
from PIL import Image

#tensorflow
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

#keras etc
import tensorflow.keras.layers
import tensorflow.keras.utils as kr_utils
import tensorflow.keras.regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

#custom
sys.path.insert(0, '../src')
from data_obj import satimg_loader
from data_fold import satimg_set

default_dataset = "mini_data"
default_splits = "mini_data"

kwa_len = len(sys.argv)
if kwa_len > 1:
    default_dataset = sys.argv[1]
if kwa_len > 2:
    default_splits = int(sys.argv[2])

img_size = 512
batch_size = 12

print("imported all")

print("making dataset...")
dataset = satimg_loader(default_dataset, default_splits, [True, True, True], [batch_size]*3,
                        img_size, "default", True, "per")

print("made datset")

b_x_trial, b_y_trial = dataset.train_fold[0][0]
print(b_x_trial.shape, b_y_trial.shape)

print("building model...")
model1 = keras.models.Sequential([keras.layers.InputLayer(input_shape=(img_size, img_size, 3)),
                                 keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2)),
                                 keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=2,activation="relu"),
                                 keras.layers.MaxPooling2D(2, 2),
                                 keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=2, activation="relu"),
                                 keras.layers.MaxPooling2D(2, 2),
                                 keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=2, activation="relu"),
                                 keras.layers.MaxPooling2D(2, 2),
                                 keras.layers.Flatten(),
                                 keras.layers.Dense(128, activation="relu"),
                                 keras.layers.Dense(128, activation="relu"),
                                 keras.layers.Dense(1)])
