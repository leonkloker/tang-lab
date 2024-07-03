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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
import sys
import pickle
import xgboost as xgb

import data
import evaluate_model

def train_model(pipeline, train, test, classification=False, weights=None, antigen="cd63"):
    # Train the model
    if not weights is None:
        pipeline = pipeline.fit(train[0], train[1], model__sample_weight=weights)
    else:
        pipeline = pipeline.fit(train[0], train[1])

    # Validate the model
    if not classification:
        if "203" in antigen:
            y_pred = np.clip(pipeline.predict(test[0]), 0, None)
        else:
            y_pred = np.clip(pipeline.predict(test[0]), 0, 1)

        metric = (evaluate_model.mae(test[1], y_pred), pearsonr(y_pred, test[1])[0])

    else:
        y_pred = pipeline.predict(test[0])
        metric = (f1_score(test[1], y_pred, average='weighted'))

    return pipeline, y_pred, metric

filename = "./results/15_fold_mean_unnormalized_freqs{}.txt".format("".join(sys.argv[1:]))
f = open(filename, "w")
sys.stdout = f

# k-fold cross validation
k = 15

# samples to use for training
samples = 9000

# Model hyperparameters
features = ["mean"] #, "min", "max", "median", "std", "q1", "q3"]

# Bin labels for classification
bins_cd203c = [0., 1.0, 14.2]
bins = [0., 0.05, 1.]

# Weights for low activation samples
weight_factor_low_activation = 1

# Features to use
ifc_features_baseline = np.arange(0, 17)
rm_freqs = [int(s) for s in sys.argv[1:]]

# Regularization parameters
alpha_lasso = 0.01
alpha_ridge = 1.0

# Define models
linear = sklin.LinearRegression()
lasso = sklin.Lasso(alpha=alpha_lasso, max_iter=10000)
ridge = sklin.Ridge(alpha=alpha_ridge, max_iter=10000)
svr = SVR(kernel='linear')
svc = SVC(kernel='linear')
xgblinear = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, objective='reg:squarederror', booster='gblinear')
xgbtree = xgb.XGBRegressor(max_depth=3, n_estimators=100, learning_rate=0.1, objective='reg:squarederror', booster='gbtree')
xgbtreec = xgb.XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.1, objective='multi:softmax', num_class=len(bins)-1)
naive_bayes = GaussianNB()
nearest_centroid = NearestCentroid()
random_forest = RandomForestClassifier(n_estimators=100, random_state=0)

# Define lists to store results
y_true_avidin = []
y_true_cd203c = []
y_true_cd63 = []
y_true_avidin_binned = []
y_true_cd203c_binned = []
y_true_cd63_binned = []
linear_y_pred_avidin = []
linear_y_pred_cd203c = []
linear_y_pred_cd63 = []
lasso_y_pred_avidin = []
lasso_y_pred_cd203c = []
lasso_y_pred_cd63 = []
ridge_y_pred_avidin = []
ridge_y_pred_cd203c = []
ridge_y_pred_cd63 = []
svr_y_pred_avidin = []
svr_y_pred_cd203c = []
svr_y_pred_cd63 = []
svc_y_pred_avidin = []
svc_y_pred_cd203c = []
svc_y_pred_cd63 = []
xgbtree_y_pred_avidin = []
xgbtree_y_pred_cd203c = []
xgbtree_y_pred_cd63 = []
xgblinear_y_pred_avidin = []
xgblinear_y_pred_cd203c = []
xgblinear_y_pred_cd63 = []
xgbtreec_y_pred_avidin = []
xgbtreec_y_pred_cd203c = []
xgbtreec_y_pred_cd63 = []
naive_bayes_y_pred_avidin = []
naive_bayes_y_pred_cd203c = []
naive_bayes_y_pred_cd63 = []
nearest_centroid_y_pred_avidin = []
nearest_centroid_y_pred_cd203c = []
nearest_centroid_y_pred_cd63 = []
random_forest_y_pred_avidin = []
random_forest_y_pred_cd203c = []
random_forest_y_pred_cd63 = []

# Define pipelines
linear_pipeline_avidin = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', linear)])
linear_pipeline_cd203c = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', linear)])
linear_pipeline_cd63 = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', linear)])

lasso_pipeline_avidin = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', lasso)])
lasso_pipeline_cd203c = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', lasso)])
lasso_pipeline_cd63 = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', lasso)])

ridge_pipeline_avidin = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', ridge)])
ridge_pipeline_cd203c = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', ridge)])
ridge_pipeline_cd63 = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', ridge)])

svr_pipeline_avidin = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', svr)])
svr_pipeline_cd203c = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', svr)])
svr_pipeline_cd63 = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', svr)])

svc_pipeline_avidin = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', svc)])
svc_pipeline_cd203c = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', svc)])
svc_pipeline_cd63 = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', svc)])

xgbtree_pipeline_avidin = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', xgbtree)])
xgbtree_pipeline_cd203c = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', xgbtree)])
xgbtree_pipeline_cd63 = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', xgbtree)])

xgblinear_pipeline_avidin = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', xgblinear)])
xgblinear_pipeline_cd203c = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', xgblinear)])
xgblinear_pipeline_cd63 = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', xgblinear)])

xgbtreec_pipeline_avidin = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', xgbtreec)])
xgbtreec_pipeline_cd203c = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', xgbtreec)])
xgbtreec_pipeline_cd63 = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', xgbtreec)])

naive_bayes_pipeline_avidin = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', naive_bayes)])
naive_bayes_pipeline_cd203c = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', naive_bayes)])
naive_bayes_pipeline_cd63 = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', naive_bayes)])

nearest_centroid_pipeline_avidin = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', nearest_centroid)])
nearest_centroid_pipeline_cd203c = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', nearest_centroid)])
nearest_centroid_pipeline_cd63 = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', nearest_centroid)])

random_forest_pipeline_avidin = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', random_forest)])
random_forest_pipeline_cd203c = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', random_forest)])
random_forest_pipeline_cd63 = sklearn.pipeline.Pipeline(steps=[('scaler', StandardScaler()), ('model', random_forest)])

for i in range(k):
    # Read in the precomputed, combined populations and all activation levels
    file_train = './data/{}_fold/{}_train_{}.pickle'.format(k, i, "_".join(features))
    file_test = './data/{}_fold/{}_test_{}.pickle'.format(k, i, "_".join(features))
    x_train, y_train_avidin, y_train_cd203c, y_train_cd63 = data.load_data(file_train)
    x_test, y_test_avidin, y_test_cd203c, y_test_cd63 = data.load_data(file_test)

    x_train = x_train[:samples]
    y_train_avidin = y_train_avidin[:samples]
    y_train_cd203c = y_train_cd203c[:samples]
    y_train_cd63 = y_train_cd63[:samples]

    # Weight the samples differently depending on activation
    weights_avidin = np.where(np.array(y_train_avidin) <= 0.1, weight_factor_low_activation, 1)
    weights_cd203c = np.where(np.array(y_train_cd203c) <= 1.4, weight_factor_low_activation, 1)
    weights_cd63 = np.where(np.array(y_train_cd63) <= 0.1, weight_factor_low_activation, 1)

    # Bin the data for classification
    y_train_avidin_binned = data.bin(y_train_avidin, bins, verbose=False)
    y_train_cd203c_binned = data.bin(y_train_cd203c, bins_cd203c, verbose=False)
    y_train_cd63_binned = data.bin(y_train_cd63, bins, verbose=False)
    y_test_avidin_binned = data.bin(y_test_avidin, bins, verbose=False)
    y_test_cd203c_binned = data.bin(y_test_cd203c, bins_cd203c, verbose=False)
    y_test_cd63_binned = data.bin(y_test_cd63, bins, verbose=False)

    # Baseline features for best performing model
    ifc_features = []
    for feature in ifc_features_baseline:
        use = True
        for rm_freq in rm_freqs:
            if feature == rm_freq-1 or feature == rm_freq+5 or (feature == rm_freq+10 and rm_freq != 1):
                use = False
                break
        if use:
            ifc_features.append(feature)
    ifc_features = np.array(ifc_features)
    x_train = x_train[:, ifc_features]
    x_test = x_test[:, ifc_features]
    
    # Train and validate models
    print("Training models using fold {}".format(i))
    _, y_pred_linear_avidin, (mae_linear_avidin, pearson_linear_avidin) = train_model(linear_pipeline_avidin, (x_train, y_train_avidin), (x_test, y_test_avidin), weights=weights_avidin, antigen="avidin")
    _, y_pred_linear_cd203c, (mae_linear_cd203c, pearson_linear_cd203c) = train_model(linear_pipeline_cd203c, (x_train, y_train_cd203c), (x_test, y_test_cd203c), weights=weights_cd203c, antigen="cd203c")
    _, y_pred_linear_cd63, (mae_linear_cd63, pearson_linear_cd63) = train_model(linear_pipeline_cd63, (x_train, y_train_cd63), (x_test, y_test_cd63), weights=weights_cd63, antigen="cd63")

    _, y_pred_lasso_avidin, (mae_lasso_avidin, pearson_lasso_avidin) = train_model(lasso_pipeline_avidin, (x_train, y_train_avidin), (x_test, y_test_avidin), weights=weights_avidin, antigen="avidin")
    _, y_pred_lasso_cd203c, (mae_lasso_cd203c, pearson_lasso_cd203c) = train_model(lasso_pipeline_cd203c, (x_train, y_train_cd203c), (x_test, y_test_cd203c), weights=weights_cd203c, antigen="cd203c")
    _, y_pred_lasso_cd63, (mae_lasso_cd63, pearson_lasso_cd63) = train_model(lasso_pipeline_cd63, (x_train, y_train_cd63), (x_test, y_test_cd63), weights=weights_cd63, antigen="cd63")

    _, y_pred_ridge_avidin, (mae_ridge_avidin, pearson_ridge_avidin) = train_model(ridge_pipeline_avidin, (x_train, y_train_avidin), (x_test, y_test_avidin), weights=weights_avidin, antigen="avidin")
    _, y_pred_ridge_cd203c, (mae_ridge_cd203c, pearson_ridge_cd203c) = train_model(ridge_pipeline_cd203c, (x_train, y_train_cd203c), (x_test, y_test_cd203c), weights=weights_cd203c, antigen="cd203c")
    _, y_pred_ridge_cd63, (mae_ridge_cd63, pearson_ridge_cd63) = train_model(ridge_pipeline_cd63, (x_train, y_train_cd63), (x_test, y_test_cd63), weights=weights_cd63, antigen="cd63")

    _, y_pred_svr_avidin, (mae_svr_avidin, pearson_svr_avidin) = train_model(svr_pipeline_avidin, (x_train, y_train_avidin), (x_test, y_test_avidin), weights=weights_avidin, antigen="avidin")
    _, y_pred_svr_cd203c, (mae_svr_cd203c, pearson_svr_cd203c) = train_model(svr_pipeline_cd203c, (x_train, y_train_cd203c), (x_test, y_test_cd203c), weights=weights_cd203c, antigen="cd203c")
    _, y_pred_svr_cd63, (mae_svr_cd63, pearson_svr_cd63) = train_model(svr_pipeline_cd63, (x_train, y_train_cd63), (x_test, y_test_cd63), weights=weights_cd63, antigen="cd63")

    _, y_pred_xgb_avidin, (mae_xgb_avidin, pearson_xgb_avidin) = train_model(xgbtree_pipeline_avidin, (x_train, y_train_avidin), (x_test, y_test_avidin), weights=weights_avidin, antigen="avidin")
    _, y_pred_xgb_cd203c, (mae_xgb_cd203c, pearson_xgb_cd203c) = train_model(xgbtree_pipeline_cd203c, (x_train, y_train_cd203c), (x_test, y_test_cd203c), weights=weights_cd203c, antigen="cd203c")
    _, y_pred_xgb_cd63, (mae_xgb_cd63, pearson_xgb_cd63) = train_model(xgbtree_pipeline_cd63, (x_train, y_train_cd63), (x_test, y_test_cd63), weights=weights_cd63, antigen="cd63")

    _, y_pred_xgblinear_avidin, (mae_xgblinear_avidin, pearson_xgblinear_avidin) = train_model(xgblinear_pipeline_avidin, (x_train, y_train_avidin), (x_test, y_test_avidin), weights=weights_avidin, antigen="avidin")
    _, y_pred_xgblinear_cd203c, (mae_xgblinear_cd203c, pearson_xgblinear_cd203c) = train_model(xgblinear_pipeline_cd203c, (x_train, y_train_cd203c), (x_test, y_test_cd203c), weights=weights_cd203c, antigen="cd203c")
    _, y_pred_xgblinear_cd63, (mae_xgblinear_cd63, pearson_xgblinear_cd63) = train_model(xgblinear_pipeline_cd63, (x_train, y_train_cd63), (x_test, y_test_cd63), weights=weights_cd63, antigen="cd63")

    _, y_pred_svc_avidin, f1_svc_avidin = train_model(svc_pipeline_avidin, (x_train, y_train_avidin_binned), (x_test, y_test_avidin_binned), classification=True, antigen="avidin")
    _, y_pred_svc_cd203c, f1_svc_cd203c = train_model(svc_pipeline_cd203c, (x_train, y_train_cd203c_binned), (x_test, y_test_cd203c_binned), classification=True, antigen="cd203c")
    _, y_pred_svc_cd63, f1_svc_cd63 = train_model(svc_pipeline_cd63, (x_train, y_train_cd63_binned), (x_test, y_test_cd63_binned), classification=True, antigen="cd63")

    _, y_pred_xgbtreec_avidin, f1_xgbtreec_avidin = train_model(xgbtreec_pipeline_avidin, (x_train, y_train_avidin_binned), (x_test, y_test_avidin_binned), classification=True, antigen="avidin")
    _, y_pred_xgbtreec_cd203c, f1_xgbtreec_cd203c = train_model(xgbtreec_pipeline_cd203c, (x_train, y_train_cd203c_binned), (x_test, y_test_cd203c_binned), classification=True, antigen="cd203c")
    _, y_pred_xgbtreec_cd63, f1_xgbtreec_cd63 = train_model(xgbtreec_pipeline_cd63, (x_train, y_train_cd63_binned), (x_test, y_test_cd63_binned), classification=True, antigen="cd63")

    _, y_pred_naive_bayes_avidin, f1_naive_bayes_avidin = train_model(naive_bayes_pipeline_avidin, (x_train, y_train_avidin_binned), (x_test, y_test_avidin_binned), classification=True, antigen="avidin")
    _, y_pred_naive_bayes_cd203c, f1_naive_bayes_cd203c = train_model(naive_bayes_pipeline_cd203c, (x_train, y_train_cd203c_binned), (x_test, y_test_cd203c_binned), classification=True, antigen="cd203c")
    _, y_pred_naive_bayes_cd63, f1_naive_bayes_cd63 = train_model(naive_bayes_pipeline_cd63, (x_train, y_train_cd63_binned), (x_test, y_test_cd63_binned), classification=True, antigen="cd63")

    _, y_pred_nearest_centroid_avidin, f1_nearest_centroid_avidin = train_model(nearest_centroid_pipeline_avidin, (x_train, y_train_avidin_binned), (x_test, y_test_avidin_binned), classification=True, antigen="avidin")
    _, y_pred_nearest_centroid_cd203c, f1_nearest_centroid_cd203c = train_model(nearest_centroid_pipeline_cd203c, (x_train, y_train_cd203c_binned), (x_test, y_test_cd203c_binned), classification=True, antigen="cd203c")
    _, y_pred_nearest_centroid_cd63, f1_nearest_centroid_cd63 = train_model(nearest_centroid_pipeline_cd63, (x_train, y_train_cd63_binned), (x_test, y_test_cd63_binned), classification=True, antigen="cd63")

    _, y_pred_random_forest_avidin, f1_random_forest_avidin = train_model(random_forest_pipeline_avidin, (x_train, y_train_avidin_binned), (x_test, y_test_avidin_binned), classification=True, antigen="avidin")
    _, y_pred_random_forest_cd203c, f1_random_forest_cd203c = train_model(random_forest_pipeline_cd203c, (x_train, y_train_cd203c_binned), (x_test, y_test_cd203c_binned), classification=True, antigen="cd203c")
    _, y_pred_random_forest_cd63, f1_random_forest_cd63 = train_model(random_forest_pipeline_cd63, (x_train, y_train_cd63_binned), (x_test, y_test_cd63_binned), classification=True, antigen="cd63")

    # Store results
    y_true_avidin.extend(y_test_avidin)
    y_true_cd203c.extend(y_test_cd203c)
    y_true_cd63.extend(y_test_cd63)
    y_true_avidin_binned.extend(y_test_avidin_binned)
    y_true_cd203c_binned.extend(y_test_cd203c_binned)
    y_true_cd63_binned.extend(y_test_cd63_binned)

    linear_y_pred_avidin.extend(y_pred_linear_avidin)
    linear_y_pred_cd203c.extend(y_pred_linear_cd203c)
    linear_y_pred_cd63.extend(y_pred_linear_cd63)
    lasso_y_pred_avidin.extend(y_pred_lasso_avidin)
    lasso_y_pred_cd203c.extend(y_pred_lasso_cd203c)
    lasso_y_pred_cd63.extend(y_pred_lasso_cd63)
    ridge_y_pred_avidin.extend(y_pred_ridge_avidin)
    ridge_y_pred_cd203c.extend(y_pred_ridge_cd203c)
    ridge_y_pred_cd63.extend(y_pred_ridge_cd63)
    svr_y_pred_avidin.extend(y_pred_svr_avidin)
    svr_y_pred_cd203c.extend(y_pred_svr_cd203c)
    svr_y_pred_cd63.extend(y_pred_svr_cd63)
    svc_y_pred_avidin.extend(y_pred_svc_avidin)
    svc_y_pred_cd203c.extend(y_pred_svc_cd203c)
    svc_y_pred_cd63.extend(y_pred_svc_cd63)
    xgbtree_y_pred_avidin.extend(y_pred_xgb_avidin)
    xgbtree_y_pred_cd203c.extend(y_pred_xgb_cd203c)
    xgbtree_y_pred_cd63.extend(y_pred_xgb_cd63)
    xgblinear_y_pred_avidin.extend(y_pred_xgblinear_avidin)
    xgblinear_y_pred_cd203c.extend(y_pred_xgblinear_cd203c)
    xgblinear_y_pred_cd63.extend(y_pred_xgblinear_cd63)
    xgbtreec_y_pred_avidin.extend(y_pred_xgbtreec_avidin)
    xgbtreec_y_pred_cd203c.extend(y_pred_xgbtreec_cd203c)
    xgbtreec_y_pred_cd63.extend(y_pred_xgbtreec_cd63)
    naive_bayes_y_pred_avidin.extend(y_pred_naive_bayes_avidin)
    naive_bayes_y_pred_cd203c.extend(y_pred_naive_bayes_cd203c)
    naive_bayes_y_pred_cd63.extend(y_pred_naive_bayes_cd63)
    nearest_centroid_y_pred_avidin.extend(y_pred_nearest_centroid_avidin)
    nearest_centroid_y_pred_cd203c.extend(y_pred_nearest_centroid_cd203c)
    nearest_centroid_y_pred_cd63.extend(y_pred_nearest_centroid_cd63)
    random_forest_y_pred_avidin.extend(y_pred_random_forest_avidin)
    random_forest_y_pred_cd203c.extend(y_pred_random_forest_cd203c)
    random_forest_y_pred_cd63.extend(y_pred_random_forest_cd63)

print("Linear Regression : ")
print("Avidin : MAE = ", evaluate_model.mae(y_true_avidin, linear_y_pred_avidin), ", Pearson correlation = ", pearsonr(linear_y_pred_avidin, y_true_avidin)[0])
print("CD203c : MAE = ", evaluate_model.mae(y_true_cd203c, linear_y_pred_cd203c), ", Pearson correlation = ", pearsonr(linear_y_pred_cd203c, y_true_cd203c)[0])
print("CD63 : MAE = ", evaluate_model.mae(y_true_cd63, linear_y_pred_cd63), ", Pearson correlation = ", pearsonr(linear_y_pred_cd63, y_true_cd63)[0])
print()

print("Lasso Regression : ")
print("Avidin : MAE = ", evaluate_model.mae(y_true_avidin, lasso_y_pred_avidin), ", Pearson correlation = ", pearsonr(lasso_y_pred_avidin, y_true_avidin)[0])
print("CD203c : MAE = ", evaluate_model.mae(y_true_cd203c, lasso_y_pred_cd203c), ", Pearson correlation = ", pearsonr(lasso_y_pred_cd203c, y_true_cd203c)[0])
print("CD63 : MAE = ", evaluate_model.mae(y_true_cd63, lasso_y_pred_cd63), ", Pearson correlation = ", pearsonr(lasso_y_pred_cd63, y_true_cd63)[0])
print()

print("Ridge Regression : ")
print("Avidin : MAE = ", evaluate_model.mae(y_true_avidin, ridge_y_pred_avidin), ", Pearson correlation = ", pearsonr(ridge_y_pred_avidin, y_true_avidin)[0])
print("CD203c : MAE = ", evaluate_model.mae(y_true_cd203c, ridge_y_pred_cd203c), ", Pearson correlation = ", pearsonr(ridge_y_pred_cd203c, y_true_cd203c)[0])
print("CD63 : MAE = ", evaluate_model.mae(y_true_cd63, ridge_y_pred_cd63), ", Pearson correlation = ", pearsonr(ridge_y_pred_cd63, y_true_cd63)[0])
print()

print("SVR : ")
print("Avidin : MAE = ", evaluate_model.mae(y_true_avidin, svr_y_pred_avidin), ", Pearson correlation = ", pearsonr(svr_y_pred_avidin, y_true_avidin)[0])
print("CD203c : MAE = ", evaluate_model.mae(y_true_cd203c, svr_y_pred_cd203c), ", Pearson correlation = ", pearsonr(svr_y_pred_cd203c, y_true_cd203c)[0])
print("CD63 : MAE = ", evaluate_model.mae(y_true_cd63, svr_y_pred_cd63), ", Pearson correlation = ", pearsonr(svr_y_pred_cd63, y_true_cd63)[0])
print()

print("XGBTree : ")
print("Avidin : MAE = ", evaluate_model.mae(y_true_avidin, xgbtree_y_pred_avidin), ", Pearson correlation = ", pearsonr(xgbtree_y_pred_avidin, y_true_avidin)[0])
print("CD203c : MAE = ", evaluate_model.mae(y_true_cd203c, xgbtree_y_pred_cd203c), ", Pearson correlation = ", pearsonr(xgbtree_y_pred_cd203c, y_true_cd203c)[0])
print("CD63 : MAE = ", evaluate_model.mae(y_true_cd63, xgbtree_y_pred_cd63), ", Pearson correlation = ", pearsonr(xgbtree_y_pred_cd63, y_true_cd63)[0])
print()

print("XGBLinear : ")
print("Avidin : MAE = ", evaluate_model.mae(y_true_avidin, xgblinear_y_pred_avidin), ", Pearson correlation = ", pearsonr(xgblinear_y_pred_avidin, y_true_avidin)[0])
print("CD203c : MAE = ", evaluate_model.mae(y_true_cd203c, xgblinear_y_pred_cd203c), ", Pearson correlation = ", pearsonr(xgblinear_y_pred_cd203c, y_true_cd203c)[0])
print("CD63 : MAE = ", evaluate_model.mae(y_true_cd63, xgblinear_y_pred_cd63), ", Pearson correlation = ", pearsonr(xgblinear_y_pred_cd63, y_true_cd63)[0])
print()

print("SVC : ")
print("Avidin : F1 = ", f1_score(y_true_avidin_binned, svc_y_pred_avidin, average='weighted'))
print("CD203c : F1 = ", f1_score(y_true_cd203c_binned, svc_y_pred_cd203c, average='weighted'))
print("CD63 : F1 = ", f1_score(y_true_cd63_binned, svc_y_pred_cd63, average='weighted'))
print()

print("XGBTree Classifier : ")
print("Avidin : F1 = ", f1_score(y_true_avidin_binned, xgbtreec_y_pred_avidin, average='weighted'))
print("CD203c : F1 = ", f1_score(y_true_cd203c_binned, xgbtreec_y_pred_cd203c, average='weighted'))
print("CD63 : F1 = ", f1_score(y_true_cd63_binned, xgbtreec_y_pred_cd63, average='weighted'))
print()

print("Naive Bayes : ")
print("Avidin : F1 = ", f1_score(y_true_avidin_binned, naive_bayes_y_pred_avidin, average='weighted'))
print("CD203c : F1 = ", f1_score(y_true_cd203c_binned, naive_bayes_y_pred_cd203c, average='weighted'))
print("CD63 : F1 = ", f1_score(y_true_cd63_binned, naive_bayes_y_pred_cd63, average='weighted'))
print()

print("Nearest Centroid : ")
print("Avidin : F1 = ", f1_score(y_true_avidin_binned, nearest_centroid_y_pred_avidin, average='weighted'))
print("CD203c : F1 = ", f1_score(y_true_cd203c_binned, nearest_centroid_y_pred_cd203c, average='weighted'))
print("CD63 : F1 = ", f1_score(y_true_cd63_binned, nearest_centroid_y_pred_cd63, average='weighted'))
print()

print("Random Forest : ")
print("Avidin : F1 = ", f1_score(y_true_avidin_binned, random_forest_y_pred_avidin, average='weighted'))
print("CD203c : F1 = ", f1_score(y_true_cd203c_binned, random_forest_y_pred_cd203c, average='weighted'))
print("CD63 : F1 = ", f1_score(y_true_cd63_binned, random_forest_y_pred_cd63, average='weighted'))
print()

f.close()

# # Plot results
# evaluate_model.plot_prediction(y_true_avidin, linear_y_pred_avidin, "./figures/moment_model/linear_regression_{}avidin_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)))
# evaluate_model.plot_prediction(y_true_cd203c, linear_y_pred_cd203c, "./figures/moment_model/linear_regression_{}cd203c_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)))
# evaluate_model.plot_prediction(y_true_cd63, linear_y_pred_cd63, "./figures/moment_model/linear_regression_{}cd63_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)))

# evaluate_model.plot_prediction(y_true_avidin, lasso_y_pred_avidin, "./figures/moment_model/lasso_regression_{}avidin_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)))
# evaluate_model.plot_prediction(y_true_cd203c, lasso_y_pred_cd203c, "./figures/moment_model/lasso_regression_{}cd203c_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)))
# evaluate_model.plot_prediction(y_true_cd63, lasso_y_pred_cd63, "./figures/moment_model/lasso_regression_{}cd63_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)))

# evaluate_model.plot_prediction(y_true_avidin, ridge_y_pred_avidin, "./figures/moment_model/ridge_regression_{}avidin_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)))
# evaluate_model.plot_prediction(y_true_cd203c, ridge_y_pred_cd203c, "./figures/moment_model/ridge_regression_{}cd203c_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)))
# evaluate_model.plot_prediction(y_true_cd63, ridge_y_pred_cd63, "./figures/moment_model/ridge_regression_{}cd63_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)))

# evaluate_model.plot_prediction(y_true_avidin, svr_y_pred_avidin, "./figures/moment_model/svr_{}avidin_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)))
# evaluate_model.plot_prediction(y_true_cd203c, svr_y_pred_cd203c, "./figures/moment_model/svr_{}cd203c_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)))
# evaluate_model.plot_prediction(y_true_cd63, svr_y_pred_cd63, "./figures/moment_model/svr_{}cd63_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)))

# evaluate_model.plot_prediction(y_true_avidin, xgbtree_y_pred_avidin, "./figures/moment_model/xgbtree_{}avidin_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)))
# evaluate_model.plot_prediction(y_true_cd203c, xgbtree_y_pred_cd203c, "./figures/moment_model/xgbtree_{}cd203c_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)))
# evaluate_model.plot_prediction(y_true_cd63, xgbtree_y_pred_cd63, "./figures/moment_model/xgbtree_{}cd63_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)))

# evaluate_model.plot_prediction(y_true_avidin, xgblinear_y_pred_avidin, "./figures/moment_model/xgblinear_{}avidin_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)))
# evaluate_model.plot_prediction(y_true_cd203c, xgblinear_y_pred_cd203c, "./figures/moment_model/xgblinear_{}cd203c_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)))
# evaluate_model.plot_prediction(y_true_cd63, xgblinear_y_pred_cd63, "./figures/moment_model/xgblinear_{}cd63_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)))

# evaluate_model.plot_confusion_matrix(y_true_avidin_binned, svc_y_pred_avidin, bins, "./figures/moment_model/sv_classifier_confusion_matrix_{}avidin_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)), labels=True)
# evaluate_model.plot_confusion_matrix(y_true_cd203c_binned, svc_y_pred_cd203c, bins_cd203c, "./figures/moment_model/sv_classifier_confusion_matrix_{}cd203c_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)), labels=True)
# evaluate_model.plot_confusion_matrix(y_true_cd63_binned, svc_y_pred_cd63, bins, "./figures/moment_model/sv_classifier_confusion_matrix_{}cd63_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)), labels=True)

# evaluate_model.plot_confusion_matrix(y_true_avidin_binned, xgbtreec_y_pred_avidin, bins, "./figures/moment_model/xgb_classifier_confusion_matrix_{}avidin_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)), labels=True)
# evaluate_model.plot_confusion_matrix(y_true_cd203c_binned, xgbtreec_y_pred_cd203c, bins_cd203c, "./figures/moment_model/xgb_classifier_confusion_matrix_{}cd203c_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)), labels=True)
# evaluate_model.plot_confusion_matrix(y_true_cd63_binned, xgbtreec_y_pred_cd63, bins, "./figures/moment_model/xgb_classifier_confusion_matrix_{}cd63_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)), labels=True)

# evaluate_model.plot_confusion_matrix(y_true_avidin_binned, naive_bayes_y_pred_avidin, bins, "./figures/moment_model/naive_bayes_classifier_confusion_matrix_{}avidin_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)), labels=True)
# evaluate_model.plot_confusion_matrix(y_true_cd203c_binned, naive_bayes_y_pred_cd203c, bins_cd203c, "./figures/moment_model/naive_bayes_classifier_confusion_matrix_{}cd203c_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)), labels=True)
# evaluate_model.plot_confusion_matrix(y_true_cd63_binned, naive_bayes_y_pred_cd63, bins, "./figures/moment_model/naive_bayes_classifier_confusion_matrix_{}cd63_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)), labels=True)

# evaluate_model.plot_confusion_matrix(y_true_avidin_binned, nearest_centroid_y_pred_avidin, bins, "./figures/moment_model/nearest_centroid_classifier_confusion_matrix_{}avidin_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)), labels=True)
# evaluate_model.plot_confusion_matrix(y_true_cd203c_binned, nearest_centroid_y_pred_cd203c, bins_cd203c, "./figures/moment_model/nearest_centroid_classifier_confusion_matrix_{}cd203c_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)), labels=True)
# evaluate_model.plot_confusion_matrix(y_true_cd63_binned, nearest_centroid_y_pred_cd63, bins, "./figures/moment_model/nearest_centroid_classifier_confusion_matrix_{}cd63_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)), labels=True)

# evaluate_model.plot_confusion_matrix(y_true_avidin_binned, random_forest_y_pred_avidin, bins, "./figures/moment_model/random_forest_classifier_confusion_matrix_{}avidin_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)), labels=True)
# evaluate_model.plot_confusion_matrix(y_true_cd203c_binned, random_forest_y_pred_cd203c, bins_cd203c, "./figures/moment_model/random_forest_classifier_confusion_matrix_{}cd203c_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)), labels=True)
# evaluate_model.plot_confusion_matrix(y_true_cd63_binned, random_forest_y_pred_cd63, bins, "./figures/moment_model/random_forest_classifier_confusion_matrix_{}cd63_{}.pdf".format("".join([str(n) for n in rm_freqs]), "_".join(features)), labels=True)
