import itertools
import joblib
import numpy as np 
import pandas as pd 
import sklearn.pipeline
import sklearn.linear_model as sklin
from sklearn.preprocessing import StandardScaler
import sys

import data
import evaluate_model

def train_model(pipeline, train, test):
    # Train the model
    pipeline = pipeline.fit(train[0], train[1])

    # Validate the model
    y_pred = pipeline.predict(test[0])
    mae = np.mean(np.abs(y_pred - test[1]))

    return pipeline, y_pred, mae

# Read in the base populations
file = './data/16_populations.pickle'
x, y = data.load_data(file)

# Get train and test populations
x_train, y_train, x_test, y_test = data.subsample_populations(x, y, split=0.8, combine_train=True, combine_test=False, max_combs=2**13)
#x_train, y_train, x_test, y_test = data.get_train_test_split(x, y, split=0.8, combine_train=True, combine_test=False)

# Get statistical moment features
features = ["mean"]#, "std", "skew", "kurt"]
x_train = data.get_statistical_moment_features(x_train, features)
x_test = data.get_statistical_moment_features(x_test, features)

# Define models
alpha_lasso = 0.001
alpha_ridge = 0.1
linear = sklin.LinearRegression()
lasso = sklin.Lasso(alpha=alpha_lasso, max_iter=10000)
ridge = sklin.Ridge(alpha=alpha_ridge, max_iter=10000)

# Define pipelines
scaler = StandardScaler()
linear_pipeline = sklearn.pipeline.Pipeline(steps=[('scaler', scaler), ('linear', linear)])
lasso_pipeline = sklearn.pipeline.Pipeline(steps=[('scaler', scaler), ('lasso', lasso)])
ridge_pipeline = sklearn.pipeline.Pipeline(steps=[('scaler', scaler), ('ridge', ridge)])

# Train and validate models
linear_pipeline, y_pred_linear, mae_linear = train_model(linear_pipeline, (x_train, y_train), (x_test, y_test))
lasso_pipeline, y_pred_lasso, mae_lasso = train_model(lasso_pipeline, (x_train, y_train), (x_test, y_test))
ridge_pipeline, y_pred_ridge, mae_ridge = train_model(ridge_pipeline, (x_train, y_train), (x_test, y_test))

# Print MAEs (R^2 not accessible as val set size is 1)
print("Linear Regression - mean absolute error: ", mae_linear)
print("Lasso Regression with alpha = {}".format(alpha_lasso),  "- mean absolute error: ", mae_lasso)
print("Ridge Regression with alpha = {}".format(alpha_ridge), "- mean absolute error: ", mae_ridge)

evaluate_model.plot_prediction(y_test, y_pred_linear, "./figures/linear/linear_regression.png")
evaluate_model.plot_prediction(y_test, y_pred_lasso, "./figures/linear/lasso_regression.png")
evaluate_model.plot_prediction(y_test, y_pred_ridge, "./figures/linear/ridge_regression.png")
#evaluate_model.plot_confusion_matrix(y[:14][:, None], y_pred[:, 2], np.linspace(0., 1., 6), "./figures/linear/ridge_regression_confusion_matrix.png")

# Train the model on the entire dataset and save it
#ridge = ridge.fit(scaler.fit_transform(x), y)
#ridge joblib.load("./models/ridge_model.pkl")
#joblib.dump(scaler, "./models/scaler.pkl")
