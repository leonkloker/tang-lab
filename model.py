import itertools
import joblib
import numpy as np 
import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score
import sklearn.pipeline
from sklearn.svm import SVC, SVR
import sklearn.linear_model as sklin
from sklearn.preprocessing import StandardScaler
import sys

import data
import evaluate_model

def train_model(pipeline, train, test, classification=False):
    # Train the model
    pipeline = pipeline.fit(train[0], train[1])

    # Validate the model
    if not classification:
        y_pred = np.clip(pipeline.predict(test[0]), 0, 1)
        metric = np.mean(np.abs(y_pred - test[1]))
    else:
        y_pred = pipeline.predict(test[0])
        metric = f1_score(test[1], y_pred, average='macro')

    return pipeline, y_pred, metric

# Read in the base populations
file = './data/16_populations.pickle'
x, y = data.load_data(file)

# Get train and test populations
x_train, y_train, x_test, y_test = data.subsample_populations(x, y, split=0.8, combine_train=True, combine_test=False, max_combs=2**13)
#x_train, y_train, x_test, y_test = data.get_train_test_split(x, y, split=0.625, combine_train=True, combine_test=False)

# Get statistical moment features
features = ["mean"]#, "std", "skew", "kurt"]
ifc_features = np.arange(0, 17)
x_train = data.get_statistical_moment_features(x_train, features)[:, ifc_features]
x_test = data.get_statistical_moment_features(x_test, features)[:, ifc_features]

# Bin labels for classification
bins = [0., 0.05, 0.20, 0.50, 1.]
y_train_binned = np.digitize(y_train, bins)
y_test_binned = np.digitize(y_test, bins)

# Define models
alpha_lasso = 0.001
alpha_ridge = 0.1
linear = sklin.LinearRegression()
lasso = sklin.Lasso(alpha=alpha_lasso, max_iter=10000)
ridge = sklin.Ridge(alpha=alpha_ridge, max_iter=10000)
svr = SVR(kernel='linear')
svc = SVC(kernel='linear')

# Define pipelines
scaler = StandardScaler()
linear_pipeline = sklearn.pipeline.Pipeline(steps=[('scaler', scaler), ('linear', linear)])
lasso_pipeline = sklearn.pipeline.Pipeline(steps=[('scaler', scaler), ('lasso', lasso)])
ridge_pipeline = sklearn.pipeline.Pipeline(steps=[('scaler', scaler), ('ridge', ridge)])
svr_pipeline = sklearn.pipeline.Pipeline(steps=[('scaler', scaler), ('svr', svr)])
svc_pipeline = sklearn.pipeline.Pipeline(steps=[('scaler', scaler), ('svr', svc)])

# Train and validate models
linear_pipeline, y_pred_linear, mae_linear = train_model(linear_pipeline, (x_train, y_train), (x_test, y_test))
lasso_pipeline, y_pred_lasso, mae_lasso = train_model(lasso_pipeline, (x_train, y_train), (x_test, y_test))
ridge_pipeline, y_pred_ridge, mae_ridge = train_model(ridge_pipeline, (x_train, y_train), (x_test, y_test))
svr_pipeline, y_pred_svr, mae_svr = train_model(svr_pipeline, (x_train, y_train), (x_test, y_test))
svc_pipeline, y_pred_svc, f1_svc = train_model(svc_pipeline, (x_train, y_train_binned), (x_test, y_test_binned), classification=True)

# Print MAEs (R^2 not accessible as val set size is 1)
print("Linear Regression - mean absolute error: ", mae_linear)
print("Lasso Regression with alpha = {}".format(alpha_lasso),  "- mean absolute error: ", mae_lasso)
print("Ridge Regression with alpha = {}".format(alpha_ridge), "- mean absolute error: ", mae_ridge)
print("Support Vector Regression - mean absolute error: ", mae_svr)
print("Support Vector Classifier - f1 score: ", f1_svc)

evaluate_model.plot_prediction(y_test, y_pred_linear, "./figures/linear/linear_regression.png")
evaluate_model.plot_prediction(y_test, y_pred_lasso, "./figures/linear/lasso_regression.png")
evaluate_model.plot_prediction(y_test, y_pred_ridge, "./figures/linear/ridge_regression.png")
evaluate_model.plot_prediction(y_test, y_pred_svr, "./figures/linear/support_vector_regression.png")

evaluate_model.plot_confusion_matrix(y_test, y_pred_linear, bins, "./figures/linear/linear_regression_confusion_matrix.png")
evaluate_model.plot_confusion_matrix(y_test, y_pred_lasso, bins, "./figures/linear/lasso_regression_confusion_matrix.png")
evaluate_model.plot_confusion_matrix(y_test, y_pred_ridge, bins, "./figures/linear/ridge_regression_confusion_matrix.png")
evaluate_model.plot_confusion_matrix(y_test, y_pred_svr, bins, "./figures/linear/sv_regression_confusion_matrix.png")
evaluate_model.plot_confusion_matrix(y_test_binned, y_pred_svc, bins, "./figures/linear/sv_classifier_confusion_matrix.png", labels=True)

# Train the model on the entire dataset and save it
#ridge = ridge.fit(scaler.fit_transform(x), y)
#ridge joblib.load("./models/ridge_model.pkl")
#joblib.dump(scaler, "./models/scaler.pkl")
