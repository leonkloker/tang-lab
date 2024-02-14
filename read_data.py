import pandas as pd
import numpy as np
import itertools
import pickle

# Read in data from a csv file and return the combined populations and their activation rates
def get_data(file, combine=False):
    features = ['demod{}_r'.format(i) for i in range(0, 6)] + ['demod{}_theta'.format(i) for i in range(0, 6)] + ['cd63+']
    df = pd.read_csv(file)[features]

    # Group by Baso population
    populations = df.groupby("cd63+")

    # Get activation percentage for each population
    y_raw = np.array(populations.mean().reset_index()["cd63+"])


    # Get cells per population
    cells_per_population = np.array(populations.size())

    if combine:
        max_populations = len(populations)
    else:
        max_populations = 1
    
    # Create all possible combinations of populations
    combined_populations = []
    combinations = []
    combined_y = []
    for k in range(1, max_populations + 1):
        for combination in itertools.combinations(range(len(populations)), k):

            # Get the activation list to know which populations are combined
            activation_list = []
            for i in combination:
                activation_list.append(y_raw[i])

            # Remember the combinations
            combinations.append(combination)

            # Combine the populations
            combined_populations.append(np.array(df[(df["cd63+"]).isin(activation_list)])[:, :-1])

            # Get the activation rate for the combined population
            combined_y.append(np.average([y_raw[i] for i in combination], weights=[cells_per_population[i] for i in combination]))

    combined_y = np.array(combined_y) / 100
    return combined_populations, combined_y, combinations

# take a list of combined populations and return the statistical moment features
def get_statistical_moment_features(combined_populations, features=["mean", "std", "skew", "kurt"]):
    # Calculate the features for each combined population
    x = []
    if "mean" in features:
        x.append(np.array([np.mean(combined_population, axis=0) for combined_population in combined_populations]))
    if "std" in features:
        x.append(np.array([np.std(combined_population, axis=0) for combined_population in combined_populations]))
    if "skew" in features:
        x.append(np.array([pd.DataFrame(combined_population).skew() for combined_population in combined_populations]))
    if "kurt" in features:
        x.append(np.array([pd.DataFrame(combined_population).kurt() for combined_population in combined_populations]))

    x = np.concatenate(x, axis=1)
    return x

# Save the features and the activation rates to a file
def save_data(file, x, y, combinations):
    with open(file, 'wb') as f:
        pickle.dump(x, f)
        pickle.dump(y, f)
        pickle.dump(combinations, f)

# Load the features and the activation rates from a file
def load_data(file):
    with open(file, 'rb') as f:
        x = pickle.load(f)
        y = pickle.load(f)
        combinations = pickle.load(f)
    return x, y, combinations

if __name__ == "__main__":
    file = 'bat_ifc.csv'
    #combined_populations, combined_y, combinations = get_data(file, combine=False)
    #features = get_statistical_moment_features(combined_populations, features=["mean", "std", "skew", "kurt"])
    features = np.load('14_populations_x.npy')
    combined_y = np.load('14_populations_y.npy')
    combinations = []
    for k in range(1, 15):
        for combination in itertools.combinations(range(14), k):
            combinations.append(combination)
    
            
    save_data('14_populations.pickle', features, combined_y, combinations)
    