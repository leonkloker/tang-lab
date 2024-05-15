import itertools
import joblib
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import random
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score
import sklearn.pipeline
from sklearn.svm import SVC, SVR
import sklearn.linear_model as sklin
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import sys
import pickle

import data
import evaluate_model

def train_model(pipeline, train, test, classification=False, weights=None, antigen="cd63"):
    # Train the model
    pipeline = pipeline.fit(train[0], train[1], model__sample_weight=weights)

    # Validate the model
    if not classification:
        if "203" in antigen:
            y_pred = np.clip(pipeline.predict(test[0]), 0, None)
        else:
            y_pred = np.clip(pipeline.predict(test[0]), 0, 1)

        metric = (np.mean(np.abs(y_pred - test[1])), pearsonr(y_pred, test[1])[0])

    else:
        y_pred = pipeline.predict(test[0])
        metric = (f1_score(test[1], y_pred, average='macro'))

    return pipeline, y_pred, metric

# Read in the base populations
antigen = "cd63" #cd63 avidin cd203c_dMFI*
file = './data/36_filtered_populations_{}.pickle'.format(antigen)
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

print("Combining the training populations...")
# Subsample the populations to get dataset
x_train_mixy, y_train_mixy, x_test_mixy, y_test_mixy = data.subsample_populations_mixy(x, y, train_split=0.99, combine_train=True, combine_test=False, max_combs=max_combs)
x_train_consty, y_train_consty, x_test_consty, y_test_consty = data.subsample_populations_consty(x, y, train_split=0.99, sample_size=0.8, combs_per_sample=int(max_combs/len(x)))
x_train_orig = [*x_train_mixy, *x_train_consty]
y_train_orig = [*y_train_mixy, *y_train_consty]
x_test_ = x_test_mixy
y_test_ = y_test_mixy

mae_ridge_list = []
mae_lasso_list = []
mae_linear_list = []
mae_svr_list = []
f1_svc_list = []
r_ridge_list = []
r_lasso_list = []
r_linear_list = []
r_svr_list = []

for n_points in np.linspace(10, 30, 30, dtype=int):

    print("Estimating the marginal distributions...")
    # Get the marginal distribution features
    query_points = data.get_query_points_marginal(x_train_orig, n_points=n_points, n_std=2)
    x_train_dist = data.get_marginal_distributions(x_train_orig, query_points)

    # Weight the samples differently depending on activation
    w = 1
    print("Weighting samples with activation <= 0.1 by a factor of {}".format(w))
    weights = np.where(np.array(y_train_orig) <= 0.1, w, 1)

    # Bin labels for classification
    if "203" in antigen:
        bins = np.linspace(0, max(y_train_orig)*1.01, 5)
    else:
        bins = [0., 0.05, 0.20, 0.50, 1.]
    y_train_binned = data.bin(y_train_orig, bins, verbose=True)
    y_test_binned = data.bin(y_test_, bins)

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

    feature_idx = []
    for feat in ifc_features:
        feature_idx = feature_idx + list(np.arange(feat * n_points, (feat + 1) * n_points, dtype=int))
    x_train_dist = x_train_dist[:,feature_idx]

    # split the data into 10 folds
    x_train_shuffled, y_train_shuffled = shuffle(x_train_dist, y_train_orig)
    n_folds = 10

    y_train_binned = data.bin(y_train_shuffled, bins, verbose=True)
    x_train_folds = np.array_split(x_train_shuffled, n_folds)
    y_train_folds = np.array_split(y_train_shuffled, n_folds)
    y_train_binned_folds = np.array_split(y_train_binned, n_folds)

    ridge_mae_list_inner = []
    linear_mae_list_inner = []
    lasso_mae_list_inner = []
    svr_mae_list_inner = []
    svc_f1_list_inner = []
    ridge_r_list_inner = []
    linear_r_list_inner = []
    lasso_r_list_inner = []
    svr_r_list_inner = []

    for i in range(n_folds):
        x_train = np.concatenate([x_train_folds[j] for j in range(n_folds) if j != i])
        y_train = np.concatenate([y_train_folds[j] for j in range(n_folds) if j != i])
        y_train_binned = np.concatenate([y_train_binned_folds[j] for j in range(n_folds) if j != i])
        x_test = x_train_folds[i]
        y_test = y_train_folds[i]
        y_test_binned = y_train_binned_folds[i]

        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, y_train_binned.shape, y_test_binned.shape)

        weights = np.where(np.array(y_train) <= 0.1, w, 1)

        # Define models
        alpha_lasso = 0.001
        alpha_ridge = 2.0
        linear = sklin.LinearRegression()
        lasso = sklin.Lasso(alpha=alpha_lasso, max_iter=10000)
        ridge = sklin.Ridge(alpha=alpha_ridge, max_iter=10000)
        svr = SVR(kernel='linear')
        svc = SVC(kernel='linear')

        # Define pipelines
        scaler = StandardScaler()
        linear_pipeline = sklearn.pipeline.Pipeline(steps=[('scaler', scaler), ('model', linear)])
        lasso_pipeline = sklearn.pipeline.Pipeline(steps=[('scaler', scaler), ('model', lasso)])
        ridge_pipeline = sklearn.pipeline.Pipeline(steps=[('scaler', scaler), ('model', ridge)])
        svr_pipeline = sklearn.pipeline.Pipeline(steps=[('scaler', scaler), ('model', svr)])
        svc_pipeline = sklearn.pipeline.Pipeline(steps=[('scaler', scaler), ('model', svc)])

        # Train and validate models
        linear_pipeline, y_pred_linear, (mae_linear, pearson_linear) = train_model(linear_pipeline, (x_train, y_train), (x_test, y_test), weights=weights, antigen=antigen)
        lasso_pipeline, y_pred_lasso, (mae_lasso, pearson_lasso) = train_model(lasso_pipeline, (x_train, y_train), (x_test, y_test), weights=weights, antigen=antigen)
        ridge_pipeline, y_pred_ridge, (mae_ridge, pearson_ridge) = train_model(ridge_pipeline, (x_train, y_train), (x_test, y_test), weights=weights, antigen=antigen)
        svr_pipeline, y_pred_svr, (mae_svr, pearson_svr) = train_model(svr_pipeline, (x_train, y_train), (x_test, y_test), weights=weights, antigen=antigen)
        svc_pipeline, y_pred_svc, (f1_svc) = train_model(svc_pipeline, (x_train, y_train_binned), (x_test, y_test_binned), classification=True, weights=weights, antigen=antigen)

        ridge_mae_list_inner.append(mae_ridge)
        linear_mae_list_inner.append(mae_linear)
        lasso_mae_list_inner.append(mae_lasso)
        svr_mae_list_inner.append(mae_svr)
        svc_f1_list_inner.append(f1_svc)
        ridge_r_list_inner.append(pearson_ridge)
        linear_r_list_inner.append(pearson_linear)
        lasso_r_list_inner.append(pearson_lasso)
        svr_r_list_inner.append(pearson_svr)
    
    mae_linear = np.mean(linear_mae_list_inner)
    mae_lasso = np.mean(lasso_mae_list_inner)
    mae_ridge = np.mean(ridge_mae_list_inner)
    mae_svr = np.mean(svr_mae_list_inner)
    f1_svc = np.mean(svc_f1_list_inner)
    pearson_linear = np.mean(linear_r_list_inner)
    pearson_lasso = np.mean(lasso_r_list_inner)
    pearson_ridge = np.mean(ridge_r_list_inner)
    pearson_svr = np.mean(svr_r_list_inner)

    print(mae_ridge, pearson_ridge)

    # Store the results
    mae_linear_list.append(mae_linear)
    mae_lasso_list.append(mae_lasso)
    mae_ridge_list.append(mae_ridge)
    mae_svr_list.append(mae_svr)
    f1_svc_list.append(f1_svc)
    r_linear_list.append(pearson_linear)
    r_lasso_list.append(pearson_lasso)
    r_ridge_list.append(pearson_ridge)
    r_svr_list.append(pearson_svr)

    print("Number of points per marginal distribution: ", n_points)


res = {
    "mae_ridge": mae_ridge_list,
    "r_ridge": r_ridge_list,
    "mae_lasso": mae_lasso_list,
    "r_lasso": r_lasso_list,
    "mae_linear": mae_linear_list,
    "r_linear": r_linear_list,
    "mae_svr": mae_svr_list,
    "r_svr": r_svr_list,
    "f1_svc": f1_svc_list
}

pickle.dump(res, open("./results/pdf_sampling_points_ablation_{}.pickle".format(antigen), "wb"))

""" plt.figure()
plt.plot(np.arange(len(linear_mae), 0, -1), linear_mae, label="Linear Regression")
plt.plot(np.arange(len(linear_mae), 0, -1), lasso_mae, label="Lasso Regression")
plt.plot(np.arange(len(linear_mae), 0, -1), ridge_mae, label="Ridge Regression")
plt.plot(np.arange(len(linear_mae), 0, -1), svr_mae, label="Support Vector Regression")
plt.legend()
plt.xlabel("Number of samples removed")
plt.ylabel("Mean absolute error")
plt.title("Model performance as a function of dataset size")
plt.grid()
plt.xlim(len(linear_mae), 0)
plt.savefig("./figures/mae_over_dataset.png")

plt.figure()
plt.plot(np.arange(len(linear_mae), 0, -1), linear_coef, label="Linear Regression")
plt.plot(np.arange(len(linear_mae), 0, -1), lasso_coef, label="Lasso Regression")
plt.plot(np.arange(len(linear_mae), 0, -1), ridge_coef, label="Ridge Regression")
plt.plot(np.arange(len(linear_mae), 0, -1), svr_coef, label="Support Vector Regression")
plt.legend()
plt.xlabel("Number of samples removed")
plt.ylabel("L2 norm of coefficients")
plt.title("Model complexity as a function of dataset size")
plt.grid()
plt.xlim(len(linear_mae), 0)
plt.savefig("./figures/coef_over_dataset.png") """

# Print metrics
print("Linear Regression : mean absolute error = ", mae_linear, ", Pearson correlation = ", pearson_linear)
print("Lasso Regression with alpha = {}".format(alpha_lasso),  ": mean absolute error = ", mae_lasso, ", Pearson correlation = ", pearson_lasso)
print("Ridge Regression with alpha = {}".format(alpha_ridge), ": mean absolute error = ", mae_ridge, ", Pearson correlation = ", pearson_ridge)
print("Support Vector Regression : mean absolute error = ", mae_svr, ", Pearson correlation = ", pearson_svr)
print("Support Vector Classifier : f1 score = ", f1_svc)

""" plt.figure()
plt.plot(n_points_list, mae_linear_list, label="Linear Regression")
plt.plot(n_points_list, mae_lasso_list, label="Lasso Regression")
plt.plot(n_points_list, mae_ridge_list, label="Ridge Regression")
plt.plot(n_points_list, mae_svr_list, label="Support Vector Regression")
plt.ylabel("Mean absolute error")
plt.xlabel("Number of points per marginal distribution")
plt.title("Model performance as a function of dataset size")
plt.legend()
plt.grid()
plt.savefig("./figures/{}_mae_over_points.png".format(antigen))

plt.figure()
plt.plot(n_points_list, f1_svc_list, label="Support Vector Classifier")
plt.ylabel("F1 score")
plt.xlabel("Number of points per marginal distribution")
plt.title("Model performance as a function of dataset size")
plt.legend()
plt.grid()
plt.savefig("./figures/{}_f1_over_points.png".format(antigen))
 """

""" evaluate_model.plot_prediction(y_test, y_pred_linear, "./figures/pdf_features/no_frequency{}/linear_regression_{}.png".format("".join([str(n) for n in rm_freqs]), antigen), classes=patients, title=mae_linear)
evaluate_model.plot_prediction(y_test, y_pred_lasso, "./figures/pdf_features/no_frequency{}/lasso_regression_{}.png".format("".join([str(n) for n in rm_freqs]),  antigen), classes=patients, title=mae_lasso)
evaluate_model.plot_prediction(y_test, y_pred_ridge, "./figures/pdf_features/no_frequency{}/ridge_regression_{}.png".format("".join([str(n) for n in rm_freqs]),  antigen), classes=patients, title=mae_ridge)
evaluate_model.plot_prediction(y_test, y_pred_svr, "./figures/pdf_features/no_frequency{}/support_vector_regression_{}.png".format("".join([str(n) for n in rm_freqs]),  antigen), classes=patients, title=mae_svr)

evaluate_model.plot_confusion_matrix(y_test, y_pred_linear, bins, "./figures/pdf_features/no_frequency{}/linear_regression_confusion_matrix_{}.png".format("".join([str(n) for n in rm_freqs]),  antigen))
evaluate_model.plot_confusion_matrix(y_test, y_pred_lasso, bins, "./figures/pdf_features/no_frequency{}/lasso_regression_confusion_matrix_{}.png".format("".join([str(n) for n in rm_freqs]),  antigen))
evaluate_model.plot_confusion_matrix(y_test, y_pred_ridge, bins, "./figures/pdf_features/no_frequency{}/ridge_regression_confusion_matrix_{}.png".format("".join([str(n) for n in rm_freqs]),  antigen))
evaluate_model.plot_confusion_matrix(y_test, y_pred_svr, bins, "./figures/pdf_features/no_frequency{}/sv_regression_confusion_matrix_{}.png".format("".join([str(n) for n in rm_freqs]),  antigen))
evaluate_model.plot_confusion_matrix(y_test_binned, y_pred_svc, bins, "./figures/pdf_features/no_frequency{}/sv_classifier_confusion_matrix_{}.png".format("".join([str(n) for n in rm_freqs]),  antigen), labels=True)
 """
# Save the results
""" file = open("./results/ablation.pickle", "wb")
pickle.dump(linear_mae, file)
pickle.dump(lasso_mae, file)
pickle.dump(ridge_mae, file)
pickle.dump(svr_mae, file)
file.close()
 """
