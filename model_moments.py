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

# Read in the precomputed, combined populations and all activation levels
file_train = './data/25_populations_16367_combinations_precomputed_trainset_antiIge_mean_min_max_median_std_q1_q3.pickle'
file_test = './data/11_populations_precomputed_testset_antiIge_mean_min_max_median_std_q1_q3.pickle'
x_train, y_train_avidin, y_train_cd203c, y_train_cd63 = data.load_data(file_train)
x_test, y_test_avidin, y_test_cd203c, y_test_cd63 = data.load_data(file_test)

samples = 16000
x_train = x_train[:samples]
y_train_avidin = y_train_avidin[:samples]
y_train_cd203c = y_train_cd203c[:samples]
y_train_cd63 = y_train_cd63[:samples]

# Weight the samples differently depending on activation
w = 1
print("Weighting samples with activation <= 0.1 by a factor of {}".format(w))
weights_avidin = np.where(np.array(y_train_avidin) <= 0.1, w, 1)
weights_cd203c = np.where(np.array(y_train_cd203c) <= 0.1, w, 1)
weights_cd63 = np.where(np.array(y_train_cd63) <= 0.1, w, 1)

# Bin labels for classification
bins_cd203c = np.linspace(0, max(y_train_cd203c)*1.01, 5)
bins = [0., 0.05, 0.20, 0.50, 1.]

print("Avidin :")
y_train_avidin_binned = data.bin(y_train_avidin, bins, verbose=True)
print("CD203c :")
y_train_cd203c_binned = data.bin(y_train_cd203c, bins_cd203c, verbose=True)
print("CD63 :")
y_train_cd63_binned = data.bin(y_train_cd63, bins, verbose=True)

y_test_avidin_binned = data.bin(y_test_avidin, bins, verbose=False)
y_test_cd203c_binned = data.bin(y_test_cd203c, bins_cd203c, verbose=False)
y_test_cd63_binned = data.bin(y_test_cd63, bins, verbose=False)

# Define which frequencies to remove
rm_freqs = []

# Baseline features for best performing model
ifc_features_baseline = np.array([0,6,7,8,9,10,11,13,14,16])
ifc_features_baseline = np.arange(0, 17)
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
print("Using features: ", ifc_features)
print()

# Remove frequencies from the marginal distribution features
""" feature_idx = []
for feat in ifc_features:
    feature_idx = feature_idx + list(np.arange(feat * n_points, (feat + 1) * n_points, dtype=int))
 """
x_train = x_train[:, ifc_features]
x_test = x_test[:, ifc_features]

# Define models
alpha_lasso = 0.001
alpha_ridge = 1.0
linear = sklin.LinearRegression()
lasso = sklin.Lasso(alpha=alpha_lasso, max_iter=10000)
ridge = sklin.Ridge(alpha=alpha_ridge, max_iter=10000)
svr = SVR(kernel='linear')
svc = SVC(kernel='linear')

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
                                              
# Train and validate models
print("Training models...")
_, y_pred_linear_avidin, (mae_linear_avidin, pearson_linear_avidin) = train_model(linear_pipeline_avidin, (x_train, y_train_avidin), (x_test, y_test_avidin), weights=weights_avidin, antigen="avidin")
_, y_pred_linear_cd203c, (mae_linear_cd203c, pearson_linear_cd203c) = train_model(linear_pipeline_cd203c, (x_train, y_train_cd203c), (x_test, y_test_cd203c), weights=weights_cd203c, antigen="cd203c")
_, y_pred_linear_cd63, (mae_linear_cd63, pearson_linear_cd63) = train_model(linear_pipeline_cd63, (x_train, y_train_cd63), (x_test, y_test_cd63), weights=weights_cd63, antigen="cd63")
print("Linear Regression : ")
print("Avidin : MAE = ", mae_linear_avidin, ", Pearson correlation = ", pearson_linear_avidin)
print("CD203c : MAE = ", mae_linear_cd203c, ", Pearson correlation = ", pearson_linear_cd203c)
print("CD63 : MAE = ", mae_linear_cd63, ", Pearson correlation = ", pearson_linear_cd63)
print()

# Train and validate models
_, y_pred_lasso_avidin, (mae_lasso_avidin, pearson_lasso_avidin) = train_model(lasso_pipeline_avidin, (x_train, y_train_avidin), (x_test, y_test_avidin), weights=weights_avidin, antigen="avidin")
_, y_pred_lasso_cd203c, (mae_lasso_cd203c, pearson_lasso_cd203c) = train_model(lasso_pipeline_cd203c, (x_train, y_train_cd203c), (x_test, y_test_cd203c), weights=weights_cd203c, antigen="cd203c")
_, y_pred_lasso_cd63, (mae_lasso_cd63, pearson_lasso_cd63) = train_model(lasso_pipeline_cd63, (x_train, y_train_cd63), (x_test, y_test_cd63), weights=weights_cd63, antigen="cd63")
print("Lasso Regression : ")
print("Avidin : MAE = ", mae_lasso_avidin, ", Pearson correlation = ", pearson_lasso_avidin)
print("CD203c : MAE = ", mae_lasso_cd203c, ", Pearson correlation = ", pearson_lasso_cd203c)
print("CD63 : MAE = ", mae_lasso_cd63, ", Pearson correlation = ", pearson_lasso_cd63)
print()

# Train and validate models
_, y_pred_ridge_avidin, (mae_ridge_avidin, pearson_ridge_avidin) = train_model(ridge_pipeline_avidin, (x_train, y_train_avidin), (x_test, y_test_avidin), weights=weights_avidin, antigen="avidin")
_, y_pred_ridge_cd203c, (mae_ridge_cd203c, pearson_ridge_cd203c) = train_model(ridge_pipeline_cd203c, (x_train, y_train_cd203c), (x_test, y_test_cd203c), weights=weights_cd203c, antigen="cd203c")
_, y_pred_ridge_cd63, (mae_ridge_cd63, pearson_ridge_cd63) = train_model(ridge_pipeline_cd63, (x_train, y_train_cd63), (x_test, y_test_cd63), weights=weights_cd63, antigen="cd63")
print("Ridge Regression : ")
print("Avidin : MAE = ", mae_ridge_avidin, ", Pearson correlation = ", pearson_ridge_avidin)
print("CD203c : MAE = ", mae_ridge_cd203c, ", Pearson correlation = ", pearson_ridge_cd203c)
print("CD63 : MAE = ", mae_ridge_cd63, ", Pearson correlation = ", pearson_ridge_cd63)
print()

# Train and validate models
_, y_pred_svr_avidin, (mae_svr_avidin, pearson_svr_avidin) = train_model(svr_pipeline_avidin, (x_train, y_train_avidin), (x_test, y_test_avidin), weights=weights_avidin, antigen="avidin")
_, y_pred_svr_cd203c, (mae_svr_cd203c, pearson_svr_cd203c) = train_model(svr_pipeline_cd203c, (x_train, y_train_cd203c), (x_test, y_test_cd203c), weights=weights_cd203c, antigen="cd203c")
_, y_pred_svr_cd63, (mae_svr_cd63, pearson_svr_cd63) = train_model(svr_pipeline_cd63, (x_train, y_train_cd63), (x_test, y_test_cd63), weights=weights_cd63, antigen="cd63")
print("Support Vector Regression : ")
print("Avidin : MAE = ", mae_svr_avidin, ", Pearson correlation = ", pearson_svr_avidin)
print("CD203c : MAE = ", mae_svr_cd203c, ", Pearson correlation = ", pearson_svr_cd203c)
print("CD63 : MAE = ", mae_svr_cd63, ", Pearson correlation = ", pearson_svr_cd63)
print()

# Train and validate models
_, y_pred_svc_avidin, f1_svc_avidin = train_model(svc_pipeline_avidin, (x_train, y_train_avidin_binned), (x_test, y_test_avidin_binned), classification=True, antigen="avidin")
_, y_pred_svc_cd203c, f1_svc_cd203c = train_model(svc_pipeline_cd203c, (x_train, y_train_cd203c_binned), (x_test, y_test_cd203c_binned), classification=True, antigen="cd203c")
_, y_pred_svc_cd63, f1_svc_cd63 = train_model(svc_pipeline_cd63, (x_train, y_train_cd63_binned), (x_test, y_test_cd63_binned), classification=True, antigen="cd63")
print("Support Vector Classifier : ")
print("Avidin : F1 score = ", f1_svc_avidin)
print("CD203c : F1 score = ", f1_svc_cd203c)
print("CD63 : F1 score = ", f1_svc_cd63)
print()

evaluate_model.plot_prediction(y_test_avidin, y_pred_linear_avidin, "./figures/pdf_features/linear_regression_{}avidin.pdf".format("".join([str(n) for n in rm_freqs])))
evaluate_model.plot_prediction(y_test_cd203c, y_pred_linear_cd203c, "./figures/pdf_features/linear_regression_{}cd203c.pdf".format("".join([str(n) for n in rm_freqs])))
evaluate_model.plot_prediction(y_test_cd63, y_pred_linear_cd63, "./figures/pdf_features/linear_regression_{}cd63.pdf".format("".join([str(n) for n in rm_freqs])))
                               
evaluate_model.plot_prediction(y_test_avidin, y_pred_lasso_avidin, "./figures/pdf_features/lasso_regression_{}avidin.pdf".format("".join([str(n) for n in rm_freqs])))
evaluate_model.plot_prediction(y_test_cd203c, y_pred_lasso_cd203c, "./figures/pdf_features/lasso_regression_{}cd203c.pdf".format("".join([str(n) for n in rm_freqs])))
evaluate_model.plot_prediction(y_test_cd63, y_pred_lasso_cd63, "./figures/pdf_features/lasso_regression_{}cd63.pdf".format("".join([str(n) for n in rm_freqs])))

evaluate_model.plot_prediction(y_test_avidin, y_pred_ridge_avidin, "./figures/pdf_features/ridge_regression_{}avidin.pdf".format("".join([str(n) for n in rm_freqs])))
evaluate_model.plot_prediction(y_test_cd203c, y_pred_ridge_cd203c, "./figures/pdf_features/ridge_regression_{}cd203c.pdf".format("".join([str(n) for n in rm_freqs])))
evaluate_model.plot_prediction(y_test_cd63, y_pred_ridge_cd63, "./figures/pdf_features/ridge_regression_{}cd63.pdf".format("".join([str(n) for n in rm_freqs])))

evaluate_model.plot_prediction(y_test_avidin, y_pred_svr_avidin, "./figures/pdf_features/svr_regression_{}avidin.pdf".format("".join([str(n) for n in rm_freqs])))
evaluate_model.plot_prediction(y_test_cd203c, y_pred_svr_cd203c, "./figures/pdf_features/svr_regression_{}cd203c.pdf".format("".join([str(n) for n in rm_freqs])))
evaluate_model.plot_prediction(y_test_cd63, y_pred_svr_cd63, "./figures/pdf_features/svr_regression_{}cd63.pdf".format("".join([str(n) for n in rm_freqs])))
                               
evaluate_model.plot_confusion_matrix(y_test_avidin_binned, y_pred_svc_avidin, bins, "./figures/pdf_features/final_model/sv_classifier_confusion_matrix_{}avidin.pdf".format("".join([str(n) for n in rm_freqs])), labels=True)
evaluate_model.plot_confusion_matrix(y_test_cd203c_binned, y_pred_svc_cd203c, bins_cd203c, "./figures/pdf_features/final_model/sv_classifier_confusion_matrix_{}cd203c.pdf".format("".join([str(n) for n in rm_freqs])), labels=True)
evaluate_model.plot_confusion_matrix(y_test_cd63_binned, y_pred_svc_cd63, bins, "./figures/pdf_features/final_model/sv_classifier_confusion_matrix_{}cd63.pdf".format("".join([str(n) for n in rm_freqs]), labels=True))
