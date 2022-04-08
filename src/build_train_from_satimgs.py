### create subset of subset based on images in sat_img

import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

mini_name = sys.argv[1]

### use to check image name lat, lon against dataset lat, lon
### to match datapoints to images
def within_eps(xlat, xlon, ylat, ylon, eps):
    if abs(xlat - ylat) <= eps and abs(xlon - ylon) <= eps:
        #within eps
        return True
    return False

if not os.path.isdir("../data/"+mini_name):
    os.mkdir("../data/"+mini_name)
    os.mkdir("../data/"+mini_name+"/test")
    os.mkdir("../data/"+mini_name+"/train_fold_0")

    rawdata = pd.read_csv("../data/csv/subsampled_fulldata.csv")
    data_cols = rawdata.columns

    raw_np = rawdata.iloc[:,:-2].to_numpy().astype('float64')
    txt_np = rawdata.iloc[:,-2:].to_numpy()
    idx_split = np.arange(raw_np.shape[0])

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
    ids_associated = []
    j_test = []
    fnames_associated = ["" for ii in range(raw_np.shape[0])]
    counts_associated = [0 for ii in range(len(sidc_fconverted))]
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
                counts_associated[j] += 1
                #print(total_matches, i, j, lat, lon, sidc_fconverted[j][2])
                ids_associated.append(i)
                j_test.append(j)
                total_matches += 1
                break
        #search for image with this lat, long
    ### record filename list
    print(" complete.")
    flinked = np.array(fnames_associated)
    print(total_matches, len(ids_associated), len(sidc_fconverted))
    #for idd in j_test:
    #    print(idd, counts_associated[idd])

    sub_ids = np.array(ids_associated)
    
    mini_numeric = raw_np[sub_ids]
    mini_text = txt_np[sub_ids]
    mini_global = idx_split[sub_ids]
    mini_imgref = flinked[sub_ids]

    df_test_num = pd.DataFrame(mini_numeric, columns = data_cols[:-2])
    df_test_txt = pd.DataFrame(mini_text, columns = data_cols[-2:])
    df_test_globalidx = pd.DataFrame(mini_global, columns=['global'])
    df_test_imgref = pd.DataFrame(mini_imgref, columns=['imgs'])
    mega_test = pd.concat([df_test_globalidx, df_test_imgref, df_test_num, df_test_txt], axis=1)
    mega_test.to_csv("../data/"+mini_name+"/test/testset.csv", index=False)
    mega_test.to_csv("../data/"+mini_name+"/train_fold_0/trainset.csv", index=False)
    mega_test.to_csv("../data/"+mini_name+"/train_fold_0/valset.csv", index=False)

else:
    print("dataset already exists in that location ("+ mini_name +")")
