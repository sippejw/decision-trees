### DATASET OBJ ATTEMPT 2

import pandas as pd
import numpy as np
import tensorflow.keras.utils as kr_utils
from PIL import Image
import math

### how to make sense of this:
### no idea
### observe_mode in {per, global, off}

class satimg_set (kr_utils.Sequence):
    def __init__ (self, data_in, shuffle, path_prefix, batch_size, expect_img_size,
                  mean_stds, dataname = "", mem_sensitive=True, observe_mode="per"):
        print("initializing datafold " + dataname)
        self.dataname = dataname
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.path_prefix = path_prefix
        self.full_data = data_in
        #expect [(3,), (3,)]
        self.mean_stds = mean_stds
        self.img_size = expect_img_size

        ### split large input into constituent components
        self.X = data_in[:, 3:-2].astype('float64')
        self.X_sup = data_in[:, -2:]
        self.X_idx = data_in[:, 0].astype('float64')
        self.X_img_ref = data_in[:, 1]
        self.y = data_in[:, 2].astype('float64').flatten()

        if self.shuffle:
            self.indexes = np.random.permutation(self.full_data.shape[0])
        else:
            self.indexes = np.arange(self.full_data.shape[0])
        self.on_epoch_end()

        self.lenn = int(np.ceil(self.full_data.shape[0] / self.batch_size))

        self.m_s_computed = False
        self.mem_sensitive = mem_sensitive

        self.observed_x = [0, 0, 0]
        self.observed_x_x = [0, 0, 0]
        self.observed_mean = [0, 0, 0]
        self.observed_std = [1, 1, 1]
        if not mem_sensitive:
            ### preload all imgs -- only if memory is not a concern
            self.img_memory = np.zeros((self.full_data.shape[0], self.img_size, self.img_size, 3))
            print("preloading images...")
            if observe_mode == "per" or observe_mode == "global":
                sum_x = np.zeros((self.img_size, self.img_size, 3))
                sum_x_x = np.zeros((self.img_size, self.img_size, 3))
                
                for i in range(self.full_data.shape[0]):
                    self.img_memory[i] = self.load_img(self.path_prefix + self.X_img_ref[i, 0])
                    #compute on here
                    #temp_img = self.load_img(self.path_prefix + self.X_img_ref[i, 0], fake_ms)
                    sum_x += self.img_memory[i]
                    sum_x_x += self.img_memory[i] * self.img_memory[i]
                print("loaded", self.full_data.shape[0].shape[0], "images.")
                for i in range(3):
                    self.observed_x[i] = np.sum(sum_x[:,:,i])
                    self.observed_x_x[i] = np.sum(sum_x_x[:,:,i])
                if mode == "per":
                    big_n = self.img_size * self.img_size * self.full_data.shape[0]
                    for i in range(3):
                        self.observed_mean[i] = self.observed_x[i]/big_n
                        self.observed_std[i] = math.sqrt((self.observed_x_x[i]/big_n) - math.pow(observed_x[i]/big_n, 2))
                elif mode == "global":
                    big_n = self.img_size * self.img_size * self.full_data.shape[0]*3
                    x_total = self.observed_x[0] + self.observed_x[1] + self.observed_x[2]
                    x_x_total = self.observed_x_x[0] + self.observed_x_x[1] + self.observed_x_x[2]
                    self.obsered_mean = [x_total/big_n] * 3
                    self.obsered_std = [math.sqrt((x_x_total/big_n) - math.pow(x_total/big_n, 2))] * 3
                print("computed observed means, std deviations")
                self.m_s_computed = True
            else:
                for i in range(self.full_data.shape[0]):
                    self.img_memory[i] = self.load_img(self.path_prefix + self.X_img_ref[i, 0])
                print("loaded", self.full_data.shape[0].shape[0], "images.")

    def compute_mean_stds(self, mode="per"):
        if self.m_s_computed:
            print("warning... means and standards have already been computed")
        if mode == "per" or mode == "global":
            fake_m_s = [np.array([0, 0, 0]), np.array([1, 1, 1])]
            temp_img = np.zeros((self.img_size, self.img_size, 3))
            sum_x = np.zeros((self.img_size, self.img_size, 3))
            sum_x_x = np.zeros((self.img_size, self.img_size, 3))
            for i in range(self.full_data.shape[0]):
                temp_img = self.load_img(self.path_prefix + self.X_img_ref[i], fake_m_s, skip=True).astype('int64')
                sum_x += temp_img
                sum_x_x += np.multiply(temp_img, temp_img)
                #print("**")
                #print(temp_img)
                #print(np.multiply(temp_img, temp_img))
                #print("**")
            for i in range(3):
                self.observed_x[i] = np.sum(sum_x[:,:,i])
                self.observed_x_x[i] = np.sum(sum_x_x[:,:,i])
            if mode == "per":
                big_n = self.img_size * self.img_size * self.full_data.shape[0]
                for i in range(3):
                    self.observed_mean[i] = self.observed_x[i]/big_n
                    print("**")
                    print(big_n)
                    print(self.observed_mean[i])
                    print(self.observed_x_x[i])
                    print(self.observed_x[i])
                    print("**")
                    self.observed_std[i] = math.sqrt((self.observed_x_x[i]/big_n) - (self.observed_x[i]/big_n)**2)
            elif mode == "global":
                big_n = self.img_size * self.img_size * self.full_data.shape[0]*3
                x_total = self.observed_x[0] + self.observed_x[1] + self.observed_x[2]
                x_x_total = self.observed_x_x[0] + self.observed_x_x[1] + self.observed_x_x[2]
                self.obsered_mean = [x_total/big_n] * 3
                self.obsered_std = [math.sqrt((x_x_total/big_n) - math.pow(x_total/big_n, 2))] * 3
            self.m_s_computed = True
        else:
            print("invalid mean/std mode")

    def get_or_compute_m_s(self, mode_in="per"):
        if self.m_s_computed:
            return [np.array(self.observed_mean), np.array(self.observed_std)]
        else:
            self.compute_mean_stds(mode=mode_in)
            return [np.array(self.observed_mean), np.array(self.observed_std)]

    def get_observed_m_s(self):
        return [np.array(self.observed_mean), np.array(self.observed_std)]

    def apply_observed_m_s(self):
        self.mean_stds = [np.array(self.observed_mean), np.array(self.observed_std)]
        if not self.mem_sensitive:
            for i in range(self.full_data.shape[0]):
                for j in range(3):
                    self.img_memory[i,:,:,j] = (self.img_memory[i,:,:,j] - self.mean_stds[0][i]) / self.mean_stds[1][i]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def load_img(self, img_loc, m_s="use_standard", skip = False):
        if m_s == "use_standard":
            m_s = self.mean_stds
        image = np.array(Image.open(img_loc))[:,:,:3]
        #print(image.shape, m_s, self.dataname)
        #if not isinstance(image, np.ndarray):
        #    if image == None:
        #        print("uhoh")
        if not skip:
            for i in range(3):
                image[:,:,i] = (image[:,:,i] - m_s[0][i]) / m_s[1][i]
        
        return image
        
    def __len__ (self):
        return self.lenn

    def __getitem__ (self, idx):
        #return picture data batch
        ret_indices = self.indexes[idx*self.batch_size : min(((idx + 1)*self.batch_size), self.full_data.shape[0])]
        if self.mem_sensitive:
            ret_imgs = np.zeros((len(ret_indices), self.img_size, self.img_size, 3))
            for i in range(len(ret_indices)):
                #print("img_ref=", self.path_prefix+self.X_img_ref[i])
                ret_imgs[i] = self.load_img(self.path_prefix + self.X_img_ref[i])
            return ret_imgs, self.y[ret_indices]
        else:
            return self.img_memory[ret_indices], self.y[ret_indices]
        
