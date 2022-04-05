### EARTHPLOTTER

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_in = pd.read_csv("../data/csv/subsampled_fulldata.csv")
latslongs = data_in.iloc[:,[1, 2, 11]]##.to_numpy().astype('float64')
latslongs.to_csv("../data/csv/mapsAPI_data.csv", index=False, float_format='%.15f')

print("done")
print("max latitude: ", np.max(data_np[:, 1]))
print("min latitude: ", np.min(data_np[:, 1]))

print("max longitude: ", np.max(data_np[:, 2]))
print("min longitude: ", np.min(data_np[:, 2]))

