import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import sys
import umap

import data

# Read in the base populations
antigen = "cd63" #cd63 avidin cd203c_dMFI*
file = './data/20_populations_train_val_{}.pickle'.format(antigen)
x, y, patients = data.load_data(file, patient_id=True)

# Shuffle the data
random.seed(2)
np.random.seed(1)
xy = list(zip(x, y))
random.shuffle(xy)
x, y = zip(*xy)
N = len(x)

# Control the dataset size
max_combs = 2**11

# Subsample the populations to get dataset
x_train_mixy, y_train_mixy, x_test_mixy, y_test_mixy = data.subsample_populations_mixy(x, y, train_split=0.5, combine_train=True, combine_test=False, max_combs=max_combs)
#x_train_consty, y_train_consty, x_test_consty, y_test_consty = data.subsample_populations_consty(x, y, train_split=0.6, sample_size=0.8, combs_per_sample=int(max_combs/len(x)))
#x_train_ = [*x_train_mixy, *x_train_consty]
#y_train = [*y_train_mixy, *y_train_consty]
x = [*x_train_mixy, *x_test_mixy]
y = [*y_train_mixy, *y_test_mixy]

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
print("Using features: ", ifc_features)

# Get the features staistical moment features
""" x_train = data.get_statistical_moment_features(x_train_, features)[:, ifc_features]
x_test = data.get_statistical_moment_features(x_test_, features)[:, ifc_features]
 """

# Get the marginal distribution features
n_points = 20
feature_idx = []
for feat in ifc_features:
    feature_idx = feature_idx + list(np.arange(feat * n_points, (feat + 1) * n_points, dtype=int))

query_points = data.get_query_points_marginal(x, n_points=n_points, n_std=2)
x = data.get_marginal_distributions(x, query_points)[:, feature_idx]

# Get the subsampled cell population as features
""" x_train, y_train = data.get_fixed_size_subsample(x_train_, y_train, size=10)
x_test, y_test = data.get_fixed_size_subsample(x_test_, y_test, size=10)
"""

# Bin labels for classification
if "203" in antigen:
    bins = np.linspace(0, max(y)*1.01, 5)
else:
    bins = [0., 0.05, 0.20, 0.50, 1.]
y_train_binned = data.bin(y, bins, verbose=True)
y_test_binned = data.bin(y, bins)


# UMAP
reducer = umap.UMAP(n_components=3, metric='euclidean', random_state=42, n_neighbors=200, min_dist=0.5)
umap_data = reducer.fit_transform(x)

# plot 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(umap_data[:, 0], umap_data[:, 1], umap_data[:, 2], cmap='plasma', s=20, c=y, alpha=0.4)
ax.set_title('3D UMAP')
colorbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
colorbar.set_label('Color scale label', rotation=270, labelpad=20)
plt.show()
plt.savefig("figures/umap_pdf_{}.png".format(antigen))


""" # plot 2d
plt.figure()
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=y, cmap='viridis', s=1, alpha=0.2)
plt.title('Data visualization using UMAP')
plt.xlabel('UMAP dimension 1')
plt.ylabel('UMAP dimension 2')
plt.colorbar(label='Class')
plt.savefig("figures/umap_2d.png") """
