import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import umap

import data

# Read in the base populations
antigen = "cd63+"
file = './data/19_populations_{}.pickle'.format(antigen)
x, y = data.load_data(file)

# Control the dataset size
max_combs = 2**10

# Subsample the populations to get dataset
x_, y_, x_test_mixy, y_test_mixy = data.subsample_populations_mixy(x, y, train_split=.99, combine_train=False, combine_test=False, max_combs=max_combs)
#x_train_consty, y_train_consty, x_test_consty, y_test_consty = data.subsample_populations_consty(x, y, train_split=.99, sample_size=0.8, combs_per_sample=int(max_combs/len(x)))

# Define statistical moment features
features = ["mean"] #, "std", "skew", "kurt"]

# Define which frequencies to remove
rm_freqs = []

# Baseline features for best performing model
ifc_features_baseline = np.array([0,6,7,8,9,10,11,13,14,16])
ifc_features = []
for feature in ifc_features_baseline:
    use = True
    for rm_freq in rm_freqs:
        if feature == rm_freq-1 or feature == rm_freq+5 or feature == rm_freq+10:
            use = False
            break
    if use:
        ifc_features.append(feature)
ifc_features = np.array(ifc_features)

# Get the features
x = data.get_statistical_moment_features(x_, features)[:, ifc_features]

scaler = StandardScaler()
x = scaler.fit_transform(x)

# classify
bins = [0., 0.05, 0.20, 0.50, 1.]
y = data.bin(y_, bins)

# UMAP
reducer = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.01, metric='euclidean', random_state=42)
data_2d = reducer.fit_transform(x)

# Plot
plt.figure()
plt.grid()
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=y, cmap='viridis', s=50)
plt.title('Data visualization using UMAP')
plt.xlabel('UMAP dimension 1')
plt.ylabel('UMAP dimension 2')
plt.colorbar(label='Class')
plt.savefig("figures/umap.png")
