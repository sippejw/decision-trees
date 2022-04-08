### model

print("importing...")
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

print("imported all")

### test
print("making dataset...")
dataset = satimg_loader("mini_data", 1, [True, True, True], [12, 12, 12],
                        512, "default", True, "per")
print("made datset")
### train is dataset.train_fold[i]
### test is dataset.test_set
### val is dataset.val_fold[i]

b_x_trial, b_y_trial = dataset.train_fold[0][0]
print(b_x_trial.shape, b_y_trial.shape)
