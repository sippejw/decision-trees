### DATA CLEANING PIPELINE PART 1
### -> load data from raw csv
### -> bin data based on fire size
### -> subsample with probability ~1/bin_freq
### -> graph firesize distribution
### -> save full subsampled data, subsampled coordinates

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

override_random_state = False
ignore_prev = False
if override_random_state:
    print("overriding previous random seed...")
    print(np.random.get_state())
else:
    with open("../rand/data_subsample.txt", "r+") as file:
        print("opening")
        prev_seed = file.read()
        #lines = file.readlines()
        print("previous seed text:", prev_seed)
        if prev_seed == "" or ignore_prev:
            print(np.random.get_state())
            print("no previous random seed")
            print("writing random state to file")
            tt = np.random.get_state()
            file.write(str(tt[0]) + ",")
            file.write(str(tt[2]) + ",")
            file.write(str(tt[3]) + ",")
            file.write(str(tt[4]) + ",")
            np.savetxt("../rand/data_subsample_np.csv", tt[1], delimiter=",")
        else:
            print("loading random state from file:")
            rs_txt = prev_seed.split(',')
            print(rs_txt)
            rsnp = np.genfromtxt('../rand/data_subsample_np.csv',delimiter=',')
            rs = (rs_txt[0], rsnp, int(rs_txt[1]), int(rs_txt[2]), float(rs_txt[3]))
            print(rs)
            np.random.set_state(rs)
    #f.close()

### Load Data from raw csv file
print("loading data...")
big_data = pd.read_csv("../data/csv/raw_data.csv")
print("column names:")
print(big_data.columns)
data_cols = big_data.columns
bdnp = big_data.to_numpy()

### Determine number of duplicates over numeric fields
"""
bdnp_num = bdnp[:,:-2].astype('float64')
dupes = bdnp.shape[0] - len(np.unique(bdnp_num, axis=0))
print(dupes)
"""

### bin data to determine sampling prob.
### n_bins set to 100,000
print('binning data...')
n_bins = 100000
pts_to_bins = np.zeros(bdnp.shape[0])
bincounts = np.zeros(n_bins)
bin_upper = big_data.FIRE_SIZE.max()
bin_lower = big_data.FIRE_SIZE.min()

### determine bins, bin counts
for i in range(bdnp.shape[0]):
    for j in range(n_bins):
        if bdnp[i, 0] <= (1+j)*(bin_upper-bin_lower)/n_bins and j < n_bins-1:
            pts_to_bins[i] = j
            bincounts[j] += 1
            break
        elif j == n_bins-1:
            pts_to_bins[i] = j
            bincounts[j] += 1
            break

### determine weights from 1/bin_freq
print("weighting bins...")
#pts_to_bins = np.array(pts_to_bins_L)
pts_bin_weights = np.zeros(bdnp.shape[0])
for i in range(bdnp.shape[0]):
    if bincounts[int(pts_to_bins[i])] != 0:
        pts_bin_weights[i] = 1/bincounts[int(pts_to_bins[i])]
    else:
        print("uhoh")
pts_bin_weights = pts_bin_weights/np.sum(pts_bin_weights)

### subsample 100,000
print("subsampling datapoints...")
indices = np.arange(bdnp.shape[0])
nsamples = 100000
ss_data = np.random.choice(indices, size=nsamples, p=pts_bin_weights)

### check distribution w/ graph
print("graphing size distribution...")
ss_sizes = np.zeros(nsamples)
for i in range(nsamples):
    ss_sizes[i] = bdnp[int(ss_data[i]), 0]

fig = plt.figure(figsize = (10, 5))
plt.hist(ss_sizes, bins=1000)
plt.yscale('log')
plt.xlabel('Fire Size (acres)')
plt.ylabel('Log Bin Frequency')
plt.title('Wildfire Size Distribution in Subsampled Dataset')
plt.savefig('../viz/subsampled_firesize_dist.png')
plt.show()

### make dataset
print("making datasets...")
ss_set = bdnp[ss_data]
df_full = pd.DataFrame(ss_set, columns = data_cols)
df_full.to_csv("../data/csv/subsampled_fulldata.csv", index=False)
#np.savetxt("../data/csv/subsampled_fulldata.csv", ss_set, delimiter=",")
np.savetxt("../data/csv/subsampled_coordinates.csv", ss_set[:,1:3], delimiter=",")
print("done")

