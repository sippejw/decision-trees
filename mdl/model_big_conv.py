#deeper cnn1

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

from tensorflow.keras.applications.efficientnet import EfficientNetB1

#custom
sys.path.insert(0, '../src')
from data_obj import satimg_loader
from data_fold import satimg_set

img_size = 512
batch_size = 12

### run with python3 model_big_conv.py dataset_name 1 n_epochs modelname True

default_dataset = "mini_data"
default_splits = 1
default_epochs = 20
default_mdlname = "best_bigconv"
save_checkpoints = False

kwa_len = len(sys.argv)
if kwa_len > 1:
    default_dataset = sys.argv[1]
if kwa_len > 2:
    default_splits = int(sys.argv[2])
if kwa_len > 3:
    default_epochs = int(sys.argv[3])
if kwa_len > 4:
    default_mdlname = sys.argv[4]
if kwa_len > 5 and sys.argv[5] == "True":
    save_checkpoints = True

img_size = 512
batch_size = 12

print("imported all")

### test
print("making dataset...")
dataset = satimg_loader("mini_data", default_splits, [True, True, True], [batch_size]*3,
                        img_size, "default", True, "per")
print("made datset")
### train is dataset.train_fold[i]
### test is dataset.test_set
### val is dataset.val_fold[i]

b_x_trial, b_y_trial = dataset.train_fold[0][0]
print(b_x_trial.shape, b_y_trial.shape)

print("building model..")

"""
base_model = EfficientNetB1(weights='imagenet', include_top=False)
img_in = keras.models.Sequential([keras.layers.InputLayer(input_shape=(img_size, img_size, 3)),
                                  leras.layers.Conv2D(fiters=3, kernel_size=(3, 3), strides=2, padding='valid', activation='relu'),
                                  keras.layers.MaxPooling2D(3, 3),
                                  leras.layers.Conv2D(fiters=3, kernel_size=(4, 4), strides=1, padding='valid', activation='relu'),
                                  keras.layers.MaxPooling2D(3, 3),
                                  leras.layers.Conv2D(fiters=3, kernel_size=(4, 4), strides=1, padding='valid', activation='relu')])

img_out = img_in.output

x = base_model.output(img_out)
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
out_pred = Dense(1, activation='relu')(x)
model = Model(inputs=base_model.input, outputs=predictions)


"""
model = keras.models.Sequential([keras.layers.InputLayer(input_shape=(img_size, img_size, 3)),
                                 keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding='same', activation="relu"),
                                 keras.layers.MaxPooling2D(2, 2),
                                 keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding='same', activation="relu"),
                                 keras.layers.MaxPooling2D(2, 2),
                                 keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=2, padding='same', activation="relu"),
                                 keras.layers.MaxPooling2D(2, 2),
                                 keras.layers.Conv2D(filters=512, kernel_size=(8,8), strides=1, activation="relu"),
                                 keras.layers.Flatten(),
                                 keras.layers.Dense(512, activation="relu"),
                                 keras.layers.Dense(256, activation="relu"),
                                 keras.layers.Dense(128, activation="relu"),
                                 keras.layers.Dense(1)])


print("model:")
print(model.summary())
print("compiling model...")
model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
### insert callbacks here
callbackL = []
if save_checkpoints:
    checkpoint_callbk = tf.keras.callbacks.ModelCheckpoint(
        default_mdlname+".h5", # name of file to save the best model to
        monitor="val_mean_squared_error", # prefix val to specify that we want the model with best macroF1 on the validation data
        verbose=1, # prints out when the model achieve a better epoch
        mode="min", # the monitored metric should be maximized
        save_freq="epoch", # clear
        save_best_only=True, # of course, if not, every time a new best is achieved will be savedf differently
        save_weights_only=True # this means that we don't have to save the architecture, if you change the architecture, you'll loose the old weights
    )
    callbackL.append(checkpoint_callbk)

print("training...")
model.fit(dataset.train_fold[0], callbacks=callbackL, epochs=default_epochs, validation_data=dataset.validation_fold[0])
print("done")
