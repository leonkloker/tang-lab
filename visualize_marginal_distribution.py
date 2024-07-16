import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import sys

plt.rcParams['font.family'] = 'Arial'

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
max_combs = 2**5

# Subsample the populations to get dataset
x_train_mixy, y_train_mixy, x_test_mixy, y_test_mixy = data.subsample_populations_mixy(x, y, train_split=0.5, combine_train=True, combine_test=False, max_combs=max_combs)
x_train_consty, y_train_consty, x_test_consty, y_test_consty = data.subsample_populations_consty(x, y, train_split=0.6, sample_size=0.8, combs_per_sample=int(max_combs/len(x)))
x = [*x_train_mixy, *x_train_consty]
y = [*y_train_mixy, *y_train_consty]

# Define statistical moment features
features = ["mean"] #, "std", "skew", "kurt"]

# Define which frequencies to remove
rm_freqs = []

# Baseline features for best performing model
ifc_features_baseline = np.array([0])
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

# Get the marginal distribution features
n_points = 100
feature_idx = []
for feat in ifc_features:
    feature_idx = feature_idx + list(np.arange(feat * n_points, (feat + 1) * n_points, dtype=int))

query_points_plot = data.get_query_points_marginal(x, n_points=n_points, n_std=3)
query_points = data.get_query_points_marginal(x, n_points=20, n_std=2)
x = data.get_marginal_distributions(x[0:2], query_points_plot)[:, :]

viridis = plt.get_cmap('viridis')
colors = [viridis(i/10) for i in range(0,10,1)]

plt.figure()
plt.plot(query_points_plot[16,:], x[0,1600:1700], color=colors[1], label="Sample distribution", linewidth=2)
plt.axvline(x=np.mean(query_points_plot[16, 49:51]), color=colors[5], linestyle='--', linewidth=2, label="Mean", zorder=10)

#for i in range(0, 20):
#    plt.axvline(x=query_points[16,i], color=colors[8], linestyle=':', linewidth=1.5)
plt.scatter(query_points[16,:], np.zeros(20), color=colors[8], s=40, label="Query points", marker='x')

plt.grid(True, linestyle=':', linewidth=0.7)
plt.xlabel('Opacity for frequency 6', fontsize=12)
plt.ylabel('Probability density', fontsize=12)

plt.legend(prop={'size': 12})
plt.tight_layout()
plt.show()
#plt.savefig("figures/marginal_pdf_freq6_opacity.png", dpi=400)

