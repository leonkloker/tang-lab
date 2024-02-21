import pandas as pd
import numpy as np
import itertools
import pickle
import scipy
import matplotlib.pyplot as plt

np.random.seed(40)

# Read in data from a csv file and return the populations and the activation rates
def get_data(file):
    features = ['demod{}_r'.format(i) for i in range(0, 6)] + ['demod{}_theta'.format(i) for i in range(0, 6)] + ['cd63+']
    df = pd.read_csv(file)[features]

    # Group by Baso population
    populations = df.groupby("cd63+")

    # Get activation percentage for each population
    y_raw = np.array(populations.mean().reset_index()["cd63+"]) / 100

    samples = []
    for _, population in populations:
        samples.append(population.iloc[:, :-1].to_numpy())

    return samples, y_raw

# Add the opacity to the populations and return the populations
def add_opacity(populations):
    opacity = []
    for population in populations:
        opacity_pop = []
        for cell in population:
            opacity_pop.append(cell[1:6] / cell[0])
        opacity.append(np.array(opacity_pop))

    for i in range(len(populations)):
        populations[i] = np.concatenate([populations[i], opacity[i]], axis=1)
    
    return populations

# Split the populations into a train and test set and return the combination of the populations and the activation rates
def get_train_test_split(populations, y_raw, split=0.8, combine_train=True, combine_test=False, max_combs=np.inf):
    n_populations = len(populations)
    train_idx = np.random.choice(n_populations, int(n_populations*split), replace=False)
    test_idx = [i for i in range(n_populations) if i not in train_idx]
    train_samples = [populations[i] for i in train_idx]
    test_samples = [populations[i] for i in test_idx]
    train_y = [y_raw[i] for i in train_idx]
    test_y = [y_raw[i] for i in test_idx]

    combined_x_train, combined_y_train, _ = combine_populations(train_samples, train_y, combine=combine_train, max_combs=max_combs)
    combined_x_test, combined_y_test, _ = combine_populations(test_samples, test_y, combine=combine_test, max_combs=max_combs)
        
    return combined_x_train, combined_y_train, combined_x_test, combined_y_test

# Subsample the populations and return the combination of the subsampled populations and the activation rates
def subsample_populations(populations, y_raw, split=0.8, combine_train=True, combine_test=False, max_combs=np.inf):
    train_samples = []
    test_samples = []
    for population in populations:
        n = population.shape[0]
        train_idx = np.random.choice(n, int(n*split), replace=False)
        test_idx = [i for i in range(n) if i not in train_idx]
        train_samples.append(population[train_idx, :])
        test_samples.append(population[test_idx, :])

    combined_x_train, combined_y_train, _ = combine_populations(train_samples, y_raw, combine=combine_train, max_combs=max_combs)
    combined_x_test, combined_y_test, _ = combine_populations(test_samples, y_raw, combine=combine_test, max_combs=max_combs)
        
    return combined_x_train, combined_y_train, combined_x_test, combined_y_test

# Combine populations and return the combined populations and the activation rates
def combine_populations(populations, y_raw, combine=True, max_combs=np.inf):
    cells_per_population = np.array([population.shape[0] for population in populations])

    if combine:
        max_populations = len(populations)
    else:
        max_populations = 1
    
    # Create all possible combinations of populations
    combined_populations = []
    combinations = []
    combined_y = []
    enough = False
    for k in range(1, max_populations + 1):
        for combination in itertools.combinations(range(len(populations)), k):

            # Remember the combinations
            combinations.append(combination)

            # Combine the populations
            combined_populations.append(np.concatenate([populations[i] for i in combination], axis=0))

            # Get the activation rate for the combined population
            combined_y.append(np.average([y_raw[i] for i in combination], weights=[cells_per_population[i] for i in combination]))

            if len(combinations) >= max_combs:
                enough = True
                break

        if enough:
            break

    combined_y = np.array(combined_y)
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

# take a list of combined populations and return the marginal estimated pdf of each feature
# evaluated at the query points
def get_marginal_distributions(features_list, query_points, method='kde'):
    results = []
    n_features = features_list[0].shape[1]

    for features in features_list:
        feature_distributions = []
        
        for i in range(n_features):
            data = features[:, i]
            kde = scipy.stats.gaussian_kde(data)
            evaluated = kde.evaluate(query_points[i])
            feature_distributions.append(evaluated)

        results.append(np.array(feature_distributions))
    return results

# Save the features and the activation rates to a file
def save_data(file, x, y, combinations=False):
    with open(file, 'wb') as f:
        pickle.dump(x, f)
        pickle.dump(y, f)
        if combinations:
            pickle.dump(combinations, f)

# Load the features and the activation rates from a file
def load_data(file, combinations=False):
    with open(file, 'rb') as f:
        x = pickle.load(f)
        y = pickle.load(f)
        if combinations:
            combinations = pickle.load(f)
            return x, y, combinations
        else:
            return x, y

if __name__ == "__main__":
    file = './data/bat_ifc.csv'
    samples, y = get_data(file)
    samples = add_opacity(samples)
    save_data('./data/16_populations.pickle', samples, y, combinations=False)

