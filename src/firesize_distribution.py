import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data stats

print("loading...")
big_data = pd.read_csv("../data/csv/raw_data.csv")
print("column names:")
print(big_data.columns)

np_sizes = big_data.FIRE_SIZE.to_numpy()
print(np_sizes.shape)
fig = plt.figure(figsize = (10, 5))
plt.hist(np_sizes, bins=1000)
plt.yscale('log')
plt.xlabel('Fire Size (acres)')
plt.ylabel('Log Bin Frequency')
plt.title('Wildfire Size Distribution in 1.88 Million US Wildfires Dataset')
plt.savefig('../viz/raw_firesize_dist.png')
plt.show()
