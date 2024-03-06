import matplotlib.pyplot as plt
import numpy as np
import random

import data

file = './data/19_populations.pickle'
x, y = data.load_data(file)

# Shuffle the data
np.random.seed(0)
xy = list(zip(x, y))
random.shuffle(xy)
x, y = zip(*xy)
N = len(x)

# Lists to store the result metrics
linear_mae = []
lasso_mae = []
ridge_mae = []
svr_mae = []
svc_f1 = []

# Control the dataset size
max_combs = 2**9

# Subsample the populations to get dataset
x_train_mixy, y_train_mixy, x_test_mixy, y_test_mixy = data.subsample_populations_mixy(x, y, train_split=0.5, combine_train=True, combine_test=False, max_combs=max_combs)
x_train_consty, y_train_consty, x_test_consty, y_test_consty = data.subsample_populations_consty(x, y, train_split=0.5, sample_size=0.8, combs_per_sample=int(max_combs/len(x)))
x_train_ = [*x_train_mixy, *x_train_consty]
y_train = [*y_train_mixy, *y_train_consty]
x_test_ = x_test_mixy
y_test = y_test_mixy

# x_train_, y_train, x_test_, y_test = data.get_train_test_split(x, y, train_split=0.5, combine_train=True, combine_test=False)

# Define statistical moment features
features = ["mean"]#, "std", "skew", "kurt"]

# Define the features to use
ifc_features = np.array([0,6,7,8,9,10,11,13,14,16])

# Get the features
x_train = data.get_statistical_moment_features(x_train_, features)[:, ifc_features]
x_test = data.get_statistical_moment_features(x_test_, features)[:, ifc_features]

X = x_train.T

U, S, V = np.linalg.svd(X, full_matrices=False)

V[0,:]

