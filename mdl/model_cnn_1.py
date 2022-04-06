### model

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
