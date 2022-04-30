# grapher
# actual vs predicted
# actual vs metric (mse, mae, elasti)

import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

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


from data_obj import satimg_loader
from data_fold import satimg_set

img_size = 512
batch_size = 1
#test_size = 0

lambduh = 0.6
print("using lamda =", lambduh, "for elasti-loss")

def elasti_loss(x, y):
    return (lambduh * tf.math.abs(x-y) + (1-lambduh) * tf.math.square(x-y))

#custom
#sys.path.insert(0, '../saved')


data_name = "mini_data"

dataset = satimg_loader("mini_data", 1, [True, True, True], [1, 1, 1],
                        img_size, "default", True, "per")

def load_bigconv(pathstr):
    bigconv = keras.models.Sequential([keras.layers.InputLayer(input_shape=(img_size, img_size, 3)),
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
    bigconv.compile(optimizer="adam", loss=elasti_loss, metrics=[elasti_loss,
                                                                 "mean_squared_error",
                                                                 "mean_absolute_error"])
    bigconv.load_weights(pathstr)
    return bigconv
    ###more depth

def load_elastiquick(pathstr):
    base_model = EfficientNetB1(weights='imagenet', include_top=False)
    img_in = keras.models.Sequential([keras.layers.InputLayer(input_shape=(img_size, img_size, 3)),
                                      keras.layers.MaxPooling2D(pool_size = (2, 2), strides=2, padding='valid')])

    #img_out = img_in.output
    print("in_shape: ", img_in.output_shape)
    print("pre_in_shape: ", base_model.input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    out_pred = Dense(1, activation='relu')(x)
    tmodel = Model(inputs=base_model.input, outputs=out_pred)

    elastiquick = Model(inputs = img_in.inputs, outputs = tmodel(img_in(img_in.inputs)))
    elastiquick.compile(optimizer="adam", loss=elasti_loss, metrics=[elasti_loss,
                                                                 "mean_squared_error",
                                                                 "mean_absolute_error"])
    for layer in base_model.layers:
        layer.trainable = False
    elastiquick.load_weights(pathstr)
    return elastiquick

def from_model(models):
    predictions = [[[], [], []] for i in range(len(models))]
    yhat = [[] for ii in range(len(models))]
    actuals = []
    for i in range(len(dataset.test_set)):
        img, lbl = dataset.test_set[i]
        actuals.append(lbl[0])
        for j in range(len(models)):
            _, elasti, mse, mae = models[j].evaluate(img, [lbl], verbose=2)
            yhat[j].append(models[j].predict(img))
            predictions[j][0].append(mse)
            predictions[j][1].append(mae)
            predictions[j][2].append(elasti)
    return actuals, yhat, predictions

def actual_predicted(data_actual, data_predicted, modelname, color):
    plt.scatter(data_actual, data_predicted,
                s=1.5, c=color, alpha=1)

    plt.xlabel("Actual Fire Size (acres)")
    plt.ylabel("Predicted Fire Size (acres)")
    plt.title("Actual vs Predicted Fire Sizes (" + modelname +", Test Set)")
    plt.show()
    plt.savefig("../saved/"+modelname+"_actual_predicted.png")

def log_actual_predicted(data_actual, data_predicted, modelname, color):
    plt.scatter(np.log(data_actual), np.log(data_predicted),
                s=1.5, c=color, alpha=1)

    plt.xlabel("Actual Fire Size (log acres)")
    plt.ylabel("Predicted Fire Size (log acres)")
    plt.title("Actual vs Predicted Fire Sizes (" + modelname +", Test Set)")
    plt.show()
    plt.savefig("../saved/"+modelname+"_log_actual_predicted.png")

def actual_metric(data_actual, metric_dist, modelname, color, metric):
    plt.scatter(data_actual, metric_dist,
                s=1.5, c=color, alpha=1)

    plt.xlabel("Actual Fire Size (acres)")
    plt.ylabel("Error (" + metric + ")")
    plt.title("Actual Fire Size Prediction Error (" + modelname +
              ", " + metric + ", Test Set)")
    plt.show()
    plt.savefig("../saved/"+modelname+"_actual_"+metric+".png")

def log_actual_metric(data_actual, metric_dist, modelname, color, metric):
    plt.scatter(np.log(data_actual), metric_dist,
                s=1.5, c=color, alpha=1)

    plt.xlabel("Log Actual Fire Size (log acres)")
    plt.ylabel("Error (" + metric + ")")
    plt.title("Actual Fire Size Prediction Error (" + modelname +
              ", " + metric + ", Test Set)")
    plt.show()
    plt.savefig("../saved/"+modelname+"_log_actual_"+metric+".png")
    

### load models
#bigconv = load_bigconv("../saved/bigconv/big_conv.dat.data-00000-of-00001")
#bigconv = load_bigconv("../mdl/modelname.h5")
elastiquick = load_elastiquick("../mdl/modelname.h5")
actuals, yhats, predictions = from_model([elastiquick])
#actuals, yhats, predictions = from_model([bigconv])
models = ["elasti"]
print(yhats)
for i in range(len(models)):
    log_actual_metric(np.array(actuals), np.array(predictions[i][2]), models[i], "orange", "elastic")
    actual_metric(np.array(actuals), np.array(predictions[i][2]), models[i], "blue", "elastic")
    actual_predicted(np.array(actuals), np.array(yhats[i]), models[i], "green")
    log_actual_predicted(np.array(actuals), np.array(yhats[i]), models[i], "red")

#log_actual_predicted([1, 2, 1, 1.5, 3], [2, 3, 3, 2, 5], "wank", "blue")
#log_actual_metric([1, 2, 1, 1.5, 3], [2, 3, 3, 2, 5], "wank", "blue", "mse")
