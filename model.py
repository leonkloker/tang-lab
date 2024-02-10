import itertools
import joblib
import numpy as np 
import pandas as pd 
import sklearn.linear_model as sklin
from sklearn.preprocessing import StandardScaler
import sys

# Read in data
df = pd.read_csv('compiled_baso_data_annotated.csv')

# Group by Baso population
populations = df.groupby("Baso population #")

# Get activation percentage for each population
y_raw = np.array(populations.mean()["% of activated basophils"])

# Get cells per population
cells_per_population = np.array(populations.size())

# Create all possible combinations of populations
combined_populations = []
combinations = []
combined_y = []
for k in range(1, len(populations) + 1):
    for combination in itertools.combinations(range(len(populations)), k):

        # Remember the combinations
        combinations.append(combination)

        # Combine the populations
        combined_populations.append(np.array(df[(df["Baso population #"] - 1).isin(combination)])[:, 1:-1])

        # Get the activation rate for the combined population
        combined_y.append(np.average([y_raw[i] for i in combination], weights=[cells_per_population[i] for i in combination]))

# Calculate the features for each combined population
mean = np.array([np.mean(combined_population, axis=0) for combined_population in combined_populations])
std = np.array([np.std(combined_population, axis=0) for combined_population in combined_populations])
skew = np.array([pd.DataFrame(combined_population).skew() for combined_population in combined_populations])
kurt = np.array([pd.DataFrame(combined_population).kurt() for combined_population in combined_populations])

# Labels to percentages
y = np.array(combined_y) / 100

# Create the feature matrix (choose which features to use)
x = np.concatenate((mean, ), axis=1)

# Define models
alpha = 0.01
linear = sklin.LinearRegression()
lasso = sklin.Lasso(alpha=alpha, max_iter=10000)
ridge = sklin.Ridge(alpha=alpha, max_iter=10000)

# K-fold cross validation (where k = number of base populations)
k = len(populations)
scores = []
mses = []
maes = []
for i in range(k):

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
    mses.append([np.mean((linear.predict(x_test) - y_test)**2), np.mean((lasso.predict(x_test) - y_test)**2), np.mean((ridge.predict(x_test) - y_test)**2)])
    maes.append([np.mean(np.abs(linear.predict(x_test) - y_test)), np.mean(np.abs(lasso.predict(x_test) - y_test)), np.mean(np.abs(ridge.predict(x_test) - y_test))])

mses = np.mean(np.array(mses), axis=0)
maes = np.mean(np.array(maes), axis=0)

# Print MAEs (R^2 not accessible as val set size is 1)
print("{}-fold cross validation score for".format(k))
print("Linear Regression - mean absolute error: ", maes[0])
print("Lasso Regression with alpha = {}".format(alpha),  "- mean absolute error: ", maes[1])
print("Ridge Regression with alpha = {}".format(alpha), "- mean absolute error: ", maes[2])

# Confusion matrix for all 2^n - 1 combinations of populations and their respective activation rates
# when using the linear model and binning the activation rates into 4 categories:
# 0 - 13%, 13 - 35%, 35 - 56%, 56 - 100%
""" bins = [0.13, 0.35, 0.56]
predictions = ridge.predict(scaler.transform(x))
predictions = np.digitize(predictions, bins)
y = np.digitize(y, bins)
confusion_matrix = np.zeros((4,4))
for i in range(len(predictions)):
    confusion_matrix[y[i]][predictions[i]] += 1
print("")
print("Confusion matrix of best model after binning predictions into 4 categories")
print("0 - 13%, 13 - 35%, 35 - 56%, 56 - 100% : ")
print(confusion_matrix) """

# Train the model on the entire dataset and save it
ridge = ridge.fit(scaler.fit_transform(x), y)
joblib.dump(ridge, "./models/ridge_model.pkl")
joblib.dump(scaler, "./models/scaler.pkl")

