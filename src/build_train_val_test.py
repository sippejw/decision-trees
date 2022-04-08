### DATA CLEANING PIPELINE PART 2
### assuming raw data is in ../data/csv/subsampled_fulldata.csv
### assuming images are in ../data/img/???
### -> document np random state
### -> make directories
### -> train/test split
### -> match samples to images
### -> save test set
### -> build train, val for each xval fold
### -> save, do graphs, stats, etc.

### eg python3 build_train_val_test.py dataset_test_1 0.3 0.2 10 true
### or python3 build_train_val_test.py dataset_test_1 0.3 0.2 1 true

import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

testmode = True

### use to check image name lat, lon against dataset lat, lon
### to match datapoints to images
def within_eps(xlat, xlon, ylat, ylon, eps):
    if abs(xlat - ylat) <= eps and abs(xlon - ylon) <= eps:
        #within eps
        return True
    return False


### deal with command line args
### diagnostics mode: use 2 to build actual dataset, 1 to test
### dataset_name, test_frac, val_frac, val_folds, diagnostics
### ie python3 build_train_val_test.py test2data 0.3 0.2 1 2
if len(sys.argv) < 5:
    print("args: dataset_name test_frac val_frac val_folds")
data_name = sys.argv[1]
test_frac = float(sys.argv[2])
val_frac = float(sys.argv[3])
val_folds = int(sys.argv[4])
if len(sys.argv) <= 5:
    diagnostics = False
elif sys.argv[5] == "0":
    diagnostics = False
    testmode = False
elif sys.argv[5] == "1":
    diagnostics = True
    testmode = True
elif sys.argv[5] == "2":
    diagnostics = True
    testmode = False
elif sys.argv[5] == "3":
    diagnostics = False
    testmode = True
    
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


### check if data already exists
### if not, make directories
if not os.path.isdir("../data/"+data_name):
    os.mkdir("../data/"+data_name)
    os.mkdir("../data/"+data_name+"/test")
    for i in range(val_folds):
        os.mkdir("../data/"+data_name+"/train_fold_" + str(i))

    ### read complete subsampled dataset
    print("building test set")
    rawdata = pd.read_csv("../data/csv/subsampled_fulldata.csv")
    data_cols = rawdata.columns
    print("column names:")
    print(rawdata.columns)
    raw_np = rawdata.iloc[:,:-2].to_numpy().astype('float64')
    txt_np = rawdata.iloc[:,-2:].to_numpy()
    idx_split = np.arange(raw_np.shape[0])

    ### compute indices for train/test split
    train_ids, test_ids, = train_test_split(idx_split, test_size=test_frac)
    test_numeric = raw_np[test_ids]
    test_txt = txt_np[test_ids]

    ### imgref section
    ### locate all image files
    ### match datapoints to images

    ### get list of all filenames in satellite_images folder
    sat_img_dir_contents = os.listdir("../data/satellite_images")
    sidc_fconverted = []

    ### filter for .png files, parse filenames into latitude and longitude floats
    for i in range(len(sat_img_dir_contents)):
        if sat_img_dir_contents[i][-4:] == ".png":
            sidc_fconverted.append(sat_img_dir_contents[i][:-4].split(',') + [sat_img_dir_contents[i]])
            sidc_fconverted[-1][0] = float(sidc_fconverted[-1][0])
            sidc_fconverted[-1][1] = float(sidc_fconverted[-1][1])
            #print(sidc_fconverted[-1])

    ### set epsilon for comparing sample lat, lon to filename lat, lon
    eps = 0.00001
    fnames_associated = ["" for ii in range(raw_np.shape[0])]
    total_matches = 0
    print("progress ", end="")

    ### check every sample against every filename
    ### if lat and lon are close, the file is correct - associate the filename with that sample
    ### keep track of number of matches... if this is less than x out of x, something is wrong
    for i in range(raw_np.shape[0]):
        if (i + 1) % raw_np.shape[0]//50 == 0:
            #progress...
            print(">", end="")
            sys.stdout.flush()
        lat = float(raw_np[i, 1])
        lon = float(raw_np[i, 2])
        for j in range(len(sidc_fconverted)):
            if within_eps(lat, lon, sidc_fconverted[j][0], sidc_fconverted[j][1], eps):
                fnames_associated[i] = sidc_fconverted[j][2]
                total_matches += 1
                break
        #search for image with this lat, long
    ### record filename list
    flinked = np.array(fnames_associated)
    print(" complete.")
    print("total matches: ", total_matches, "out of", raw_np.shape[0])
    
    ###
    ###
    ###

    ### build dataframe for test data, save test data
    df_test_num = pd.DataFrame(test_numeric, columns = data_cols[:-2])
    df_test_txt = pd.DataFrame(test_txt, columns = data_cols[-2:])
    df_test_globalidx = pd.DataFrame(test_ids, columns=['global'])
    df_test_imgref = pd.DataFrame(flinked[test_ids], columns=['imgs'])
    mega_test = pd.concat([df_test_globalidx, df_test_imgref, df_test_num, df_test_txt], axis=1)
    mega_test.to_csv("../data/"+data_name+"/test/testset.csv", index=False)

    ### histogram to compare test and combined train set
    ### also do some additional stats?
    if diagnostics:
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
        if not testmode:
            plt.show()

    ### generate crossvalidation folds
    for i in range(val_folds):
        print("building validation fold " + str(val_folds))
        train_i, val_i = train_test_split(train_ids, test_size=val_frac)
        train_i_num = raw_np[train_i]
        train_i_txt = txt_np[train_i]
        val_i_num = raw_np[val_i]
        val_i_txt = txt_np[val_i]

        ### build fold-i train set, save it
        df_train_num = pd.DataFrame(train_i_num, columns = data_cols[:-2])
        df_train_txt = pd.DataFrame(train_i_txt, columns = data_cols[-2:])
        df_train_globalidx = pd.DataFrame(train_i, columns=['global'])
        df_train_imgref = pd.DataFrame(flinked[train_i], columns=['imgs'])
        mega_train = pd.concat([df_train_globalidx, df_train_imgref, df_train_num, df_train_txt], axis=1)
        mega_train.to_csv("../data/"+data_name+"/train_fold_"+str(i)+"/trainset.csv", index=False)

        ### build fold-i val set, save it
        df_val_num = pd.DataFrame(val_i_num, columns = data_cols[:-2])
        df_val_txt = pd.DataFrame(val_i_txt, columns = data_cols[-2:])
        df_val_globalidx = pd.DataFrame(val_i, columns=['global'])
        df_val_imgref = pd.DataFrame(flinked[val_i], columns=['imgs'])
        mega_val = pd.concat([df_val_globalidx, df_val_imgref, df_val_num, df_val_txt], axis=1)
        mega_val.to_csv("../data/"+data_name+"/train_fold_"+str(i)+"/valset.csv", index=False)

        ### histogram to compare fold-i train and val sets
        ### also do some additional stats?
        if diagnostics:
            fig = plt.figure(figsize = (10, 5))
            plt.hist(train_i_num[:,0].flatten(), bins=500, align='left', rwidth=0.5)
            plt.hist(val_i_num[:,0].flatten(), bins=500, align='right', rwidth=0.5, color='red')
            plt.yscale('log')
            plt.xlabel('Fire Size (acres)')
            plt.ylabel('Log Bin Frequency')
            plt.title('Wildfire Size Distribution of Train and Validation Datasets, fold ' + str(i))
            plt.savefig('../viz/train_val_firesize_dist_fold_'+str(i)+'.png')
            if not testmode:
                plt.show()

    ### clean up if needed
    if testmode:
        if data_name != "csv" and data_name != "satellite_images":
            os.system('rm -r ../data/'+data_name)
        
    print("done")
else:
    print("dataset name already taken")
