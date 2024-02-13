import joblib
import numpy as np
import pandas as pd
import sklearn.linear_model as sklin

import read_data

# Load models
linear = joblib.load('./models/ridge_model.pkl')
scaler = joblib.load('./models/scaler.pkl')

# Read in data
file = 'bat_ifc.csv'
combined_populations, combined_y, combinations = read_data.get_data(file, combine=True)
features = read_data.get_statistical_moment_features(combined_populations, features=["mean"])

print(features.shape)

# Group by Baso population

# Get activation percentage for each population

# Caclulate the features

# Scale the features

# Predict the activation percentage
