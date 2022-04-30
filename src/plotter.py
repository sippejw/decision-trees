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


### SETUP DATA
### change this based on the model
### gp_name, mdl_paths, mdl_names, mdl_loaders must be the same size
### load this dataset
data_name = "mini_data"
### directory where graphs are spat out
group_name = "model_comparison_1"
### locations of models to load
model_paths = ["../mdl/modelname.h5", "../mdl/mdlwank.h5"]
### names of models
#model_names = ["ENET-B1-Modified-Elastic"] #also add "Big-Conv-IFS-MSE"
model_names = ["ENET-B1-Modified-Elastic", "Big-Conv-IFS-MSE"]
### model loaders (so keras knows how to load the weights)
### ...see down below, fuck you python
### store loaded models here
loaded_models = []
saveloc = ""
#keep
metric = ["MSE", "MAE", "ELASTIC (lambda = 0.6)"]
#colors for graphs... must be same size, 4
actual_colors = ["blue", "orange", "pink", "grey"]
log_colors = ["green", "red", "purple", "black"]
dotsize = 3





dataset = satimg_loader(data_name, 1, [True, True, True], [1, 1, 1],
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
    #print("in_shape: ", img_in.output_shape)
    #print("pre_in_shape: ", base_model.input_shape)
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

def from_model(models, verbosity=0):
    predictions = [[[], [], []] for i in range(len(models))]
    yhat = [[] for ii in range(len(models))]
    actuals = []
    for i in range(len(dataset.test_set)):
        img, lbl = dataset.test_set[i]
        actuals.append(lbl[0])
        for j in range(len(models)):
            _, elasti, mse, mae = models[j].evaluate(img, [lbl], verbose=verbosity)
            yhat[j].append(models[j].predict(img))
            predictions[j][0].append(mse)
            predictions[j][1].append(mae)
            predictions[j][2].append(elasti)
    return actuals, yhat, predictions

def actual_predicted(data_actual, data_predicted, modelname, color):
    plt.scatter(data_actual, data_predicted,
                s=dotsize, c=color, alpha=1)

    plt.xlabel("Actual Fire Size (acres)")
    plt.ylabel("Predicted Fire Size (acres)")
    plt.title("Actual vs Predicted Fire Sizes \n(" + modelname +", Test Set)")
    plt.savefig(saveloc + "/" + modelname + "_actual_predicted.png")
    plt.clf()
    #plt.show()

def log_actual_predicted(data_actual, data_predicted, modelname, color):
    plt.scatter(np.log(data_actual), np.log(data_predicted),
                s=dotsize, c=color, alpha=1)

    plt.xlabel("Actual Fire Size (log acres)")
    plt.ylabel("Predicted Fire Size (log acres)")
    plt.title("Actual vs Predicted Fire Sizes \n(" + modelname +", Test Set)")
    plt.savefig(saveloc + "/" + modelname + "_log_actual_predicted.png")
    plt.clf()
    #plt.show()

def actual_metric(data_actual, metric_dist, modelname, color, metric):
    plt.scatter(data_actual, metric_dist,
                s=dotsize, c=color, alpha=1)

    plt.xlabel("Actual Fire Size (acres)")
    plt.ylabel("Error (" + metric + ")")
    plt.title("Actual Fire Size Prediction Error \n(" + modelname +
              ", " + metric + ", Test Set)")
    plt.savefig(saveloc + "/" + modelname + "_actual_"+metric+".png")
    plt.clf()
    #plt.show()

def log_actual_metric(data_actual, metric_dist, modelname, color, metric):
    plt.scatter(np.log(data_actual), metric_dist,
                s=dotsize, c=color, alpha=1)

    plt.xlabel("Log Actual Fire Size (log acres)")
    plt.ylabel("Error (" + metric + ")")
    plt.title("Actual Fire Size Prediction Error \n(" + modelname +
              ", " + metric + ", Test Set)")
    plt.savefig(saveloc + "/" + modelname + "_log_actual_"+metric+".png")
    plt.clf()
    #plt.show()

def n_girls_1_pyplt(data_x, data_y, logx, logy, mdlnames, colors, metric, xlabel, ylabel, title, drawlines=True):
    if logx:
        for i in range(len(data_x)):
            data_x[i] = np.log(data_x[i])
    if logy:
        for i in range(len(data_y)):
            data_y[i] = np.log(data_y[i])
    if drawlines:
        for i in range(len(data_x[0])):
            xv = [data_x[0][i], data_x[0][i]]
            yv = [data_y[0][i], data_y[1][i]]
            plt.plot(xv, yv, c="black", linewidth=0.5, alpha=0.7)
    for i in range(len(data_x)):
        plt.scatter(data_x[i], data_y[i], s=dotsize, c=colors[i], label=mdlnames[i])
    
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title("Model Comparison (" + title + ")")
    plt.savefig(saveloc + "/mdl_comparison_"+title+".png")
    plt.clf()
    



#model_loaders = [load_elastiquick]
model_loaders = [load_elastiquick, load_bigconv]

### load models
#bigconv = load_bigconv("../saved/bigconv/big_conv.dat.data-00000-of-00001")
#bigconv = load_bigconv("../mdl/modelname.h5")
os.system("rm -r ../saved/comparison_" + group_name)
os.mkdir("../saved/comparison_" + group_name)
saveloc =  "../saved/comparison_" + group_name
for i in range(len(model_names)):
    loaded_models.append(model_loaders[i](model_paths[i]))
actuals, yhats, predictions = from_model(loaded_models)
for i in range(len(model_names)):
    actual_predicted(np.array(actuals), np.array(yhats[i]), model_names[i], actual_colors[0])
    log_actual_predicted(np.array(actuals), np.array(yhats[i]), model_names[i], log_colors[0])

    for j in range(len(metric)):
        actual_metric(np.array(actuals), np.array(predictions[i][j]), model_names[i], actual_colors[j+1], metric[j])
        log_actual_metric(np.array(actuals), np.array(predictions[i][j]), model_names[i], log_colors[j+1], metric[j])

print(np.array(yhats).shape)
print(np.array(yhats)[0][0])
n_girls_1_pyplt([np.array(actuals), np.array(actuals)],
                [np.array(yhats[0])[:,0], np.array(yhats[1])[:,0]],
                logx=True, logy=True, mdlnames = model_names,
                colors=actual_colors, metric="",
                xlabel="Actual Log Fire Size (log acres)",
                ylabel="Predicted Log Fire Size (log acres)",
                title="Predicted Fire Size")
#also include overlay??
#p


print("DATA REPORT ***")
for i in range(len(model_names)):
    #compute avg metrics
    print("- MODEL " + model_names[i])
    for j in range(len(metric)):
        avg_metric = sum(predictions[i][j])/len(predictions[i][j])
        print("  - METRIC " + metric[j] + ": " + str(avg_metric))
        
    #print(model_names[i], "penis")
    
"""
elastiquick = load_elastiquick("../mdl/modelname.h5")
actuals, yhats, predictions = from_model([elastiquick])
#actuals, yhats, predictions = from_model([bigconv])
#models = ["elasti"]
print(yhats)
for i in range(len(models)):
    log_actual_metric(np.array(actuals), np.array(predictions[i][2]), models[i], "orange", "elastic")
    actual_metric(np.array(actuals), np.array(predictions[i][2]), models[i], "blue", "elastic")
    actual_predicted(np.array(actuals), np.array(yhats[i]), models[i], "green")
    log_actual_predicted(np.array(actuals), np.array(yhats[i]), models[i], "red")

#log_actual_predicted([1, 2, 1, 1.5, 3], [2, 3, 3, 2, 5], "wank", "blue")
#log_actual_metric([1, 2, 1, 1.5, 3], [2, 3, 3, 2, 5], "wank", "blue", "mse")
"""
