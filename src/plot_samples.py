import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

full_set = np.genfromtxt('../data/csv/fire_sizes.csv')
subsample = np.genfromtxt('../data/csv/subsampled_fulldata.csv')

print(full_set)
fig = plt.figure(figsize = (10, 5))
plt.hist(full_set, bins=500, align='left', rwidth=0.5)
plt.hist(subsample[:,0].flatten(), bins=500, align='right', rwidth=0.5, color='red')
plt.yscale('log')
plt.xlabel('Fire Size (acres)')
plt.ylabel('Log Bin Frequency')
plt.title('Wildifre Size Distributions in subsample and full dataset')
plt.savefig('../viz/full_train_test_firesize_dist.png')
