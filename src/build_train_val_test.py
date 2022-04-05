### DATA CLEANING PIPELINE PART 2
### assuming raw data is in ../data/csv/subsampled_fulldata.csv
### assuming images are in ../data/img/???
### -> split into train/test/val

### eg python3 build_train_val_test.py dataset_test_1 0.3 0.2 10 true

import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

### dataset_name, test_frac, val_frac, val_folds, diagnostics
if len(sys.argv) < 5:
    print("args: dataset_name test_frac val_frac val_folds")
data_name = sys.argv[1]
test_frac = float(sys.argv[2])
val_frac = float(sys.argv[3])
val_folds = int(sys.argv[4])
if len(sys.argv) <= 5:
    diagnostics = False
elif sys.argv[5] == "true":
    diagnostics = True

### deal with random states
override_random_state = False
ignore_prev = True
if override_random_state:
    print("overriding previous random seed...")
    print(np.random.get_state())
else:
    with open("../rand/split_" + data_name + ".txt", "a+") as file:
        #print("opening")
        prev_seed = file.read()
        #lines = file.readlines()
        #print("previous seed text:", prev_seed)
        if ignore_prev and diagnostics:
            print(np.random.get_state())
            init_state = np.random.get_state()
            print("writing random state to file")
            tt = np.random.get_state()
            file.write(str(tt[0]) + "," + str(tt[2]) + ",")
            file.write(str(tt[3]) + "," + str(tt[4]) + ",")
            np.savetxt("../rand/split_" + data_name + "_np.csv", tt[1], delimiter=",")
        elif diagnostics:
            print("loading random state from file:")
            rs_txt = prev_seed.split(',')
            print(rs_txt)
            rsnp = np.genfromtxt("../rand/split_" + data_name + "_np.csv",delimiter=',')
            rs = (rs_txt[0], rsnp, int(rs_txt[1]), int(rs_txt[2]), float(rs_txt[3]))
            print(rs)
            np.random.set_state(rs)


if not os.path.isdir("../data/"+data_name):
    os.mkdir("../data/"+data_name)
    os.mkdir("../data/"+data_name+"/test")
    for i in range(val_folds):
        os.mkdir("../data/"+data_name+"/train_fold_" + str(i))

    print("building test set")
    rawdata = pd.read_csv("../data/csv/subsampled_fulldata.csv")
    data_cols = rawdata.columns
    print("column names:")
    print(rawdata.columns)
    raw_np = rawdata.iloc[:,:-2].to_numpy().astype('float64')
    txt_np = rawdata.iloc[:,-2:].to_numpy()
    idx_split = np.arange(raw_np.shape[0])
    train_ids, test_ids, = train_test_split(idx_split, test_size=test_frac)
    test_numeric = raw_np[test_ids]
    test_txt = txt_np[test_ids]

    ###imgref
    for i in range(raw_np.shape[0]):
        lat = raw_np[i, 1]
        lon = raw_np[i, 2]
        #search for image with this lat, long
        

    ### save test data
    df_test_num = pd.DataFrame(test_numeric, columns = data_cols[:-2])
    df_test_txt = pd.DataFrame(test_txt, columns = data_cols[-2:])
    df_test_globalidx = pd.DataFrame(test_ids, columns=['global'])
    df_test_imgref = pd.DataFrame(test_ids, columns=['global'])
    mega_test = pd.concat([df_test_globalidx, df_test_num, df_test_txt], axis=1)
    mega_test.to_csv("../data/"+data_name+"/test/testset.csv", index=False)
                          
    if diagnostics:
        #do diagnostics
        big_train_numeric = raw_np[train_ids]
        big_train_txt = txt_np[train_ids]

        fig = plt.figure(figsize = (10, 5))
        plt.hist(big_train_numeric[:,0].flatten(), bins=500, align='left', rwidth=0.5)
        plt.hist(test_numeric[:,0].flatten(), bins=500, align='right', rwidth=0.5, color='red')
        plt.yscale('log')
        plt.xlabel('Fire Size (acres)')
        plt.ylabel('Log Bin Frequency')
        plt.title('Wildfire Size Distributions in Test and Complete Train Datasets')
        plt.savefig('../viz/full_train_test_firesize_dist.png')
        plt.show()
        
    for i in range(val_folds):
        print("building validation fold " + str(val_folds))
        train_i, val_i = train_test_split(train_ids, test_size=val_frac)
        train_i_num = raw_np[train_i]
        train_i_txt = txt_np[train_i]
        val_i_num = raw_np[val_i]
        val_i_txt = txt_np[val_i]
        
        df_train_num = pd.DataFrame(train_i_num, columns = data_cols[:-2])
        df_train_txt = pd.DataFrame(train_i_txt, columns = data_cols[-2:])
        df_train_globalidx = pd.DataFrame(train_i, columns=['global'])
        mega_train = pd.concat([df_train_globalidx, df_train_num, df_train_txt], axis=1)
        mega_train.to_csv("../data/"+data_name+"/train_fold_"+str(i)+"/trainset.csv", index=False)

        df_val_num = pd.DataFrame(val_i_num, columns = data_cols[:-2])
        df_val_txt = pd.DataFrame(val_i_txt, columns = data_cols[-2:])
        df_val_globalidx = pd.DataFrame(val_i, columns=['global'])
        mega_val = pd.concat([df_val_globalidx, df_val_num, df_val_txt], axis=1)
        mega_val.to_csv("../data/"+data_name+"/train_fold_"+str(i)+"/valset.csv", index=False)
        if diagnostics:
            fig = plt.figure(figsize = (10, 5))
            plt.hist(train_i_num[:,0].flatten(), bins=500, align='left', rwidth=0.5)
            plt.hist(val_i_num[:,0].flatten(), bins=500, align='right', rwidth=0.5, color='red')
            plt.yscale('log')
            plt.xlabel('Fire Size (acres)')
            plt.ylabel('Log Bin Frequency')
            plt.title('Wildfire Size Distribution of Train and Validation Datasets, fold ' + str(i))
            plt.savefig('../viz/train_val_firesize_dist_fold_'+str(i)+'.png')
            plt.show()
        
    print("done")
else:
    print("dataset name already taken")
