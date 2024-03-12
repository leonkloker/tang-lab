import itertools
import joblib
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import random
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score
import sklearn.pipeline
from sklearn.svm import SVC, SVR
import sklearn.linear_model as sklin
from sklearn.preprocessing import StandardScaler
import sys
import pickle

import data
import evaluate_model

def train_model(pipeline, train, test, classification=False, weights=None):
    # Train the model
    pipeline = pipeline.fit(train[0], train[1], model__sample_weight=weights)

    # Validate the model
    if not classification:
        y_pred = np.clip(pipeline.predict(test[0]), 0, 1)
        metric = np.mean(np.abs(y_pred - test[1]))
    else:
        y_pred = pipeline.predict(test[0])
        metric = f1_score(test[1], y_pred, average='macro')

    return pipeline, y_pred, metric

# Read in the base populations
antigen = "cd63+"
file = './data/19_populations_{}.pickle'.format(antigen)
x, y = data.load_data(file)

# Shuffle the data
random.seed(2)
np.random.seed(1)
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
linear_coef = []
lasso_coef = []
ridge_coef = []
svr_coef = []
svc_coef = []

# Control the dataset size
max_combs = 2**13

for n in range(N-11, 0, -1):
    print(n)
    # Subsample the populations to get dataset
    x_train_mixy, y_train_mixy, x_test_mixy, y_test_mixy = data.subsample_populations_mixy(x[:-n], y[:-n], train_split=0.5, combine_train=True, combine_test=False, max_combs=max_combs)
    x_train_consty, y_train_consty, x_test_consty, y_test_consty = data.subsample_populations_consty(x[:-n], y[:-n], train_split=0.6, sample_size=0.8, combs_per_sample=int(max_combs/len(x)))
    x_train_ = [*x_train_mixy, *x_train_consty]
    y_train = [*y_train_mixy, *y_train_consty]
    x_test_ = [*x_test_mixy, *x[-n:]]
    y_test = [*y_test_mixy, *y[-n:]]

    # x_train_, y_train, x_test_, y_test = data.get_train_test_split(x, y, train_split=0.5, combine_train=True, combine_test=False)

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

    # Get the features
    x_train = data.get_statistical_moment_features(x_train_, features)[:, ifc_features]
    x_test = data.get_statistical_moment_features(x_test_, features)[:, ifc_features]

    # PCA of feature matrix, use coefficients as feature
    coef = np.array([])
    if len(coef) > 0:
        U, S, V = np.linalg.svd(x_train.T, full_matrices=False)
        pca_features_train = V[coef,:].T
        pca_features_test = (np.diag(1/S) @ U.T @ x_test.T)[coef,:].T

        # Add PCA features to the dataset
        x_train = np.column_stack((x_train, pca_features_train))
        x_test = np.column_stack((x_test, pca_features_test))

    # Weight the samples differently depending on activation
    w = 1
    print("Weighting samples with activation <= 0.1 by a factor of {}".format(w))
    weights = np.where(np.array(y_train) <= 0.1, w, 1)

    # Bin labels for classification
    bins = [0., 0.05, 0.20, 0.50, 1.]
    y_train_binned = data.bin(y_train, bins, verbose=True)
    y_test_binned = data.bin(y_test, bins)

    # Define models
    alpha_lasso = 0.001
    alpha_ridge = 1
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
    linear_pipeline, y_pred_linear, mae_linear = train_model(linear_pipeline, (x_train, y_train), (x_test, y_test), weights=weights)
    lasso_pipeline, y_pred_lasso, mae_lasso = train_model(lasso_pipeline, (x_train, y_train), (x_test, y_test), weights=weights)
    ridge_pipeline, y_pred_ridge, mae_ridge = train_model(ridge_pipeline, (x_train, y_train), (x_test, y_test), weights=weights)
    svr_pipeline, y_pred_svr, mae_svr = train_model(svr_pipeline, (x_train, y_train), (x_test, y_test), weights=weights)
    svc_pipeline, y_pred_svc, f1_svc = train_model(svc_pipeline, (x_train, y_train_binned), (x_test, y_test_binned), classification=True, weights=weights)

    linear_mae.append(mae_linear)
    lasso_mae.append(mae_lasso)
    ridge_mae.append(mae_ridge)
    svr_mae.append(mae_svr)
    svc_f1.append(f1_svc)

    linear_coef.append(np.linalg.norm(linear_pipeline.named_steps['model'].coef_))
    lasso_coef.append(np.linalg.norm(lasso_pipeline.named_steps['model'].coef_))
    ridge_coef.append(np.linalg.norm(ridge_pipeline.named_steps['model'].coef_))
    svr_coef.append(np.linalg.norm(svr_pipeline.named_steps['model'].coef_))
    svc_coef.append(np.linalg.norm(svc_pipeline.named_steps['model'].coef_))

plt.figure()
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
plt.savefig("./figures/coef_over_dataset.png")


""" # Print MAEs (R^2 not accessible as val set size is 1)
print("Linear Regression : mean absolute error = ", mae_linear)
print("Lasso Regression with alpha = {}".format(alpha_lasso),  ": mean absolute error = ", mae_lasso)
print("Ridge Regression with alpha = {}".format(alpha_ridge), ": mean absolute error = ", mae_ridge)
print("Support Vector Regression : mean absolute error = ", mae_svr)
print("Support Vector Classifier : f1 score = ", f1_svc)
 """
""" evaluate_model.plot_prediction(y_test, y_pred_linear, "./figures/models/no_frequency{}/linear_regression_{}.png".format(rm_freq, antigen), title=mae_linear)
evaluate_model.plot_prediction(y_test, y_pred_lasso, "./figures/models/no_frequency{}/lasso_regression_{}.png".format(rm_freq, antigen), title=mae_lasso)
evaluate_model.plot_prediction(y_test, y_pred_ridge, "./figures/models/no_frequency{}/ridge_regression_{}.png".format(rm_freq, antigen), title=mae_ridge)
evaluate_model.plot_prediction(y_test, y_pred_svr, "./figures/models/no_frequency{}/support_vector_regression_{}.png".format(rm_freq, antigen), title=mae_svr)

evaluate_model.plot_confusion_matrix(y_test, y_pred_linear, bins, "./figures/models/no_frequency{}/linear_regression_confusion_matrix_{}.png".format(rm_freq, antigen))
evaluate_model.plot_confusion_matrix(y_test, y_pred_lasso, bins, "./figures/models/no_frequency{}/lasso_regression_confusion_matrix_{}.png".format(rm_freq, antigen))
evaluate_model.plot_confusion_matrix(y_test, y_pred_ridge, bins, "./figures/models/no_frequency{}/ridge_regression_confusion_matrix_{}.png".format(rm_freq, antigen))
evaluate_model.plot_confusion_matrix(y_test, y_pred_svr, bins, "./figures/models/no_frequency{}/sv_regression_confusion_matrix_{}.png".format(rm_freq, antigen))
evaluate_model.plot_confusion_matrix(y_test_binned, y_pred_svc, bins, "./figures/models/no_frequency{}/sv_classifier_confusion_matrix_{}.png".format(rm_freq, antigen), labels=True)
 """
# Save the results
""" file = open("./results/ablation.pickle", "wb")
pickle.dump(linear_mae, file)
pickle.dump(lasso_mae, file)
pickle.dump(ridge_mae, file)
pickle.dump(svr_mae, file)
file.close()
 """
# Train the model on the entire dataset and save it
#ridge = ridge.fit(scaler.fit_transform(x), y)
#ridge joblib.load("./models/ridge_model.pkl")
#joblib.dump(scaler, "./models/scaler.pkl")
