### DATASET LOADER

import pandas as pd
import numpy as np
from data_fold import satimg_set

class satimg_loader:
    ### dataname:       name of dataset file            string
    ### expect_folds:   number of x-validation folds    int
    ### shuffle:        whether to shuffle each epoch   [bool, bool, bool]
    ### batch:          batch sizes for each set        [int, int, int]
    ### imgsize:        dimensions of image expected    int
    ### musigs:         mean, std deviations to use     ...
    ### mem:            memory sensitive mode           bool
    ### omode:          whether to use per-fold or global means, stds
    
    def __init__ (self, dataname, expect_folds, shuffle, batch, imgsize, musigs, mem, omode):
        print("building sets...")
        self.dataset_name = dataname
        self.k_folds = expect_folds
        #self.test_params = test_params
        #self.val_params = val_params
        #self.train_params = train_params

        ### LOAD IN THE DATA
        ### COLS:
        ### IDX | IMGref | FIRE_SIZE | LAT | LONG | YEAR | DISC_DATE | CONT_DATE | DIC_TIME | DISC_DOY | CONT_DOY |
        ### CONT_TIME | STAT_CAUSE | FIRE_NAME

        if musigs == "default":
            musigs = [[np.array([0, 0, 0]), np.array([1, 1, 1])]]*3

        ### TODO: fix normalization for test mode, eg. do normailization on combined train/val

        test_data_in_np = pd.read_csv("../data/"+dataname+"/test/testset.csv").to_numpy()
        self.test_set = satimg_set(test_data_in_np, shuffle[0], "../data/satellite_images/", batch[0], imgsize,
                              musigs[0], dataname=dataname + "_test_set", mem_sensitive=mem, observe_mode=omode)

        self.train_fold = []
        self.validation_fold = []

        for i in range(self.k_folds):
            temp_train = pd.read_csv("../data/"+dataname+"/train_fold_"+str(i)+"/trainset.csv").to_numpy()
            temp_val = pd.read_csv("../data/"+dataname+"/train_fold_"+str(i)+"/trainset.csv").to_numpy()

            self.train_fold.append(satimg_set(temp_train, shuffle[2], "../data/satellite_images/", batch[2], imgsize,
                              musigs[2], dataname=dataname + "_train_set", mem_sensitive=mem, observe_mode=omode))
            fold_m_s = self.train_fold[-1].get_or_compute_m_s(mode_in=omode)
            self.train_fold[-1].apply_observed_m_s()

            self.validation_fold.append(satimg_set(temp_val, shuffle[1], "../data/satellite_images/", batch[1], imgsize,
                              fold_m_s, dataname=dataname + "_validation_set", mem_sensitive=mem, observe_mode=omode))

    
        print("sets built.")
        """
            
            
        X_test = __test_data_in.iloc[:,3:-2].to_numpy().astype('float64')
        X_test_sup = __test_data_in.iloc[:,-2:].to_numpy()
        X_test_idx = __test_data_in.iloc[:,0].to_numpy().astype('float64')
        X_test_img_ref = __test_data_in.iloc[:,1].to_numpy()
        y_test = __test_data_in.iloc[:,0].to_numpy().astype('float64').flatten()
        
        X_train= []
        X_valid = []
        X_train_sup = []
        X_valid_sup = []
        X_train_idx = []
        X_valid_idx = []
        X_train_img_ref = []
        X_valid_img_ref = []
        y_train = []
        y_valid = []
        
        for i in range(self.k_folds):
            temp_train = pd.read_csv("../data/"+dataname+"/train_fold_"+str(i)+"/trainset.csv")
            temp_val = pd.read_csv("../data/"+dataname+"/train_fold_"+str(i)+"/trainset.csv")

            X_train.append(temp_train.iloc[:,3:-2].to_numpy().astype('float64'))
            X_valid.append(temp_val.iloc[:,3:-2].to_numpy().astype('float64'))
            
            X_train_sup.append(temp_train.iloc[:,-2:].to_numpy())
            X_valid_sup.append(temp_val.iloc[:,-2:].to_numpy())

            X_train_idx.append(temp_train.iloc[:,0].to_numpy().astype('float64'))
            X_valid_idx.append(temp_val.iloc[:,0].to_numpy().astype('float64'))

            X_train_img_ref.append(temp_train.iloc[:,1].to_numpy())
            X_valid_img_ref.append(temp_val.iloc[:,1].to_numpy())

            y_train.append(temp_train.iloc[:,0].to_numpy().astype('float64').flatten())
            y_valid.append(temp_val.iloc[:,0].to_numpy().astype('float64').flatten())
            
        if self.k_folds == 1:
            X_train= X_train[0]
            X_valid = X_valid[0]
            X_train_sup = X_train_sup[0]
            X_valid_sup = X_valid_sup[0]
            X_train_idx = X_train_idx[0]
            X_valid_idx = X_valid_idx[0]
            y_train = y_train[0]
            y_valid = y_valid[0]
        print("loaded data successfully")
        #__train_data = pd.read_csv("../data/"+dataname+"/test/testset.csv")
    def setmode (self, mode):
        if mode == "train" or mode == "test" or mode = "val":
            self.mode = mode
        else:
            print("invalid mode set")
            
    def __len__ (self):
        return len
        """
        
