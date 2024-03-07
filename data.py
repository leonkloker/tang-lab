import pandas as pd
import numpy as np
import itertools
import pickle
import scipy
import matplotlib.pyplot as plt

# Read in data from a csv file and return the populations and the activation rates
def get_data(file, antigen="cd63+"):
    features = ['demod{}_r'.format(i) for i in range(0, 6)] + ['demod{}_theta'.format(i) for i in range(0, 6)] + [antigen]
    df = pd.read_csv(file)[features]

    # Group by Baso population
    populations = df.groupby(antigen)

    if antigen == "cd203c_norm_diff":
        denominator = 1
    else:
        denominator = 100

    # Get activation percentage for each population
    y_raw = np.array(populations.mean().reset_index()[antigen]) / denominator

    samples = []
    for i, (_, population) in enumerate(populations):
        if y_raw[i] > 1:
            continue
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
def get_train_test_split(populations, y_raw, train_split=0.8, combine_train=True, combine_test=False, max_combs=np.inf):
    n_populations = len(populations)
    train_idx = np.random.choice(n_populations, int(n_populations*train_split), replace=False)
    test_idx = [i for i in range(n_populations) if i not in train_idx]
    train_samples = [populations[i] for i in train_idx]
    test_samples = [populations[i] for i in test_idx]
    train_y = [y_raw[i] for i in train_idx]
    test_y = [y_raw[i] for i in test_idx]

    combined_x_train, combined_y_train, _ = combine_populations(train_samples, train_y, combine=combine_train, max_combs=max_combs)
    combined_x_test, combined_y_test, _ = combine_populations(test_samples, test_y, combine=combine_test, max_combs=max_combs)
        
    return combined_x_train, combined_y_train, combined_x_test, combined_y_test

# Subsample the populations and return the combination of the subsampled populations and the activation rates
# Here, each base population is split randomly into a train and test set
# If combine_train is True, the train sets are combined into all possible combinations
# If combine_test is True, the test sets are combined into all possible combinations
def subsample_populations_mixy(populations, y_raw, train_split=0.8, combine_train=True, combine_test=False, max_combs=np.inf):
    train_samples = []
    test_samples = []
    for population in populations:
        n = population.shape[0]
        train_idx = np.random.choice(n, int(n*train_split), replace=False)
        test_idx = [i for i in range(n) if i not in train_idx]
        train_samples.append(population[train_idx, :])
        test_samples.append(population[test_idx, :])

    combined_x_train, combined_y_train, _ = combine_populations(train_samples, y_raw, combine=combine_train, max_combs=max_combs)
    combined_x_test, combined_y_test, _ = combine_populations(test_samples, y_raw, combine=combine_test, max_combs=max_combs)
        
    return combined_x_train, combined_y_train, combined_x_test, combined_y_test

# Subsample the populations and return the combination of the subsampled populations and the activation rates
# Here, each base population of size n is split randomly into a train and test set
# the train and test sets are resampled to yield size n * train_split * sample_size
# Hence, base populations are not mixed
def subsample_populations_consty(populations, y_raw, train_split=0.8, sample_size=0.7, combs_per_sample=2**12):
    train_samples = []
    test_samples = []
    y_train = []
    y_test = []
    for population, y in zip(populations, y_raw):
        n = population.shape[0]
        train_idx = np.random.choice(n, int(n*train_split), replace=False)
        test_idx = [i for i in range(n) if i not in train_idx]

        for i in range(combs_per_sample):
            train_idx_sub = np.random.choice(train_idx, int(len(train_idx)*sample_size), replace=False)
            train_samples.append(population[train_idx_sub, :])
            y_train.append(y)

        test_samples.append(population[test_idx, :])
        y_test.append(y)
        
    return train_samples, y_train, test_samples, y_test

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

def bin(y, bins, verbose=False):
    y = np.digitize(y, bins)
    if verbose:
        for i in range(len(bins)-1):
            print("{} samples with {} < y <= {}".format(np.sum(y == i+1), bins[i], bins[i+1]))
    return y

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
    antigen = "cd203c_norm_diff"
    file = './data/bat_ifc.csv'
    samples, y = get_data(file, antigen=antigen)
    samples = add_opacity(samples)
    save_data('./data/19_populations_{}.pickle'.format(antigen), samples, y, combinations=False)

