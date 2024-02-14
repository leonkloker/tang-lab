import itertools
import joblib
import numpy as np 
import pandas as pd 
import sklearn.linear_model as sklin
from sklearn.preprocessing import StandardScaler
import sys

import read_data
import evaluate_model

# Read in data
file = '14_populations.pickle'
x, y, combinations = read_data.load_data(file)

# Use only the mean as a feature
x = x[:, :12]

# Define models
alpha = 0.1
linear = sklin.LinearRegression()
lasso = sklin.Lasso(alpha=alpha, max_iter=10000)
ridge = sklin.Ridge(alpha=alpha, max_iter=10000)

# K-fold cross validation (where k = number of base populations)
k = max(combinations[-1])
scores = []
mses = []
maes = []
y_pred = []
for i in range(k+1):

    # Get indices for training (all combinations of populations that don't contain the current one)
    train_indices = [j for j in range(len(combinations)) if i not in combinations[j]]
    test_indices = [i]

    # Get training data
    x_train = x[train_indices]
    y_train = y[train_indices]

    # Get test data (current population without any combinations to avoid data leakage)
    x_test = x[test_indices]
    y_test = y[test_indices]

    # Scale data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Fit models
    linear = linear.fit(x_train, y_train)
    lasso = lasso.fit(x_train, y_train)
    ridge = ridge.fit(x_train, y_train)

    # Validate models
    y_pred.append([linear.predict(x_test), lasso.predict(x_test), ridge.predict(x_test)])
    mses.append([np.mean((linear.predict(x_test) - y_test)**2), np.mean((lasso.predict(x_test) - y_test)**2), np.mean((ridge.predict(x_test) - y_test)**2)])
    maes.append([np.mean(np.abs(linear.predict(x_test) - y_test)), np.mean(np.abs(lasso.predict(x_test) - y_test)), np.mean(np.abs(ridge.predict(x_test) - y_test))])

mses = np.mean(np.array(mses), axis=0)
maes = np.mean(np.array(maes), axis=0)
y_pred = np.array(y_pred)

# Print MAEs (R^2 not accessible as val set size is 1)
print("{}-fold cross validation score for".format(k))
print("Linear Regression - mean absolute error: ", maes[0])
print("Lasso Regression with alpha = {}".format(alpha),  "- mean absolute error: ", maes[1])
print("Ridge Regression with alpha = {}".format(alpha), "- mean absolute error: ", maes[2])

evaluate_model.plot_prediction(y[0:14][:, None], y_pred[:, 0], "./figures/linear/linear_regression_cross_validation.png")
#evaluate_model.plot_prediction(y[0:14][:, None], y_pred[:, 1], "./figures/linear/lasso_regression_cross_validation.png")
#evaluate_model.plot_prediction(y[0:14][:, None], y_pred[:, 2], "./figures/linear/ridge_regression_cross_validation.png")

#evaluate_model.plot_confusion_matrix(y[:14][:, None], y_pred[:, 2], np.linspace(0., 1., 6), "./figures/linear/ridge_regression_confusion_matrix.png")

# Train the model on the entire dataset and save it
#ridge = ridge.fit(scaler.fit_transform(x), y)
ridge joblib.load("./models/ridge_model.pkl")
#joblib.dump(scaler, "./models/scaler.pkl")
