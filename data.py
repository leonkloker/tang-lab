import pandas as pd
import numpy as np
import itertools
import pickle
import scipy
import sys
import matplotlib.pyplot as plt
import random
import os
from torch.utils.data import Dataset
from sklearn.model_selection import KFold

# Read in data from a csv file and return the populations and the activation rates
def get_data(file, normalize_negative_control=False):
    ifc_features = ['demod{}_r'.format(i) for i in range(0, 6)] + ['demod{}_theta'.format(i) for i in range(0, 6)]
    features = ifc_features + ["cd63"] + ["cd203c_dMFI*"] + ["avidin"] + ['patient_id'] + ['date'] + ["dose"]
    df = pd.read_csv(file)[features]

    # remove the experiments from date 04/24/2024
    df = df[df['date'] != '04/24/2024']
    df = df.drop(columns=['date'])

    # remove potential outliers

    # Group by Baso population
    populations = df.groupby(["cd63", 'patient_id'])

    patient_ids = sorted(set(populations.mean().reset_index()['patient_id']))
    patient_ids_samples = []
    samples = []
    y_raw_avidin = []
    y_raw_cd203c = []
    y_raw_cd63 = []
    
    # Get the negative control for each patient
    for patient in patient_ids:
        control = np.array(df[(df['patient_id'] == patient) & (df['dose'] == 0)][ifc_features])
        control = add_opacity([control])[0]
        control = np.mean(control, axis=0)

        # Find the populations for the patient
        for i, (_, population) in enumerate(populations):
            if population['patient_id'].iloc[0] == patient:
                if np.any(np.array(population["dose"]) != 0) or not normalize_negative_control:
                    y_raw_avidin.append(population["avidin"].mean() / 100)
                    y_raw_cd203c.append(population["cd203c_dMFI*"].mean())
                    y_raw_cd63.append(population["cd63"].mean() / 100)
                    population_features = np.array(population[ifc_features])
                    population_features = add_opacity([population_features])[0]

                    # Normalize the population by the negative control
                    if normalize_negative_control:
                        population_features[:, 0:6] = population_features[:, 0:6] / control[0:6]
                        population_features[:, 6:12] = population_features[:, 6:12] - control[6:12]
                        population_features[:, 12:] = population_features[:, 12:] / control[12:]

                    samples.append(population_features)
                    patient_ids_samples.append(patient)

    return samples, y_raw_avidin, y_raw_cd203c, y_raw_cd63, patient_ids_samples

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
            train_idx_sub = np.random.choice(train_idx, int(len(train_idx)*sample_size), replace=True)
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
    done = False

    for k in range(1, max_populations + 1):
        for combination in itertools.combinations(range(len(populations)), k):
            combinations.append(combination)
    
            # Combine the populations
            combined_populations.append(np.concatenate([populations[i] for i in combination], axis=0))

            # Get the activation rate for the combined population
            combined_y.append(np.average([y_raw[i] for i in combination], weights=[cells_per_population[i] for i in combination]))

            if len(combinations) >= max_combs:
                done = True
                break
        if done:
            break

    combined_y = np.array(combined_y)
    return combined_populations, combined_y, combinations

# take a list of combined populations and return the statistical moment features
def get_statistical_moment_features(combined_populations, features=["mean", "std", "skew", "kurt", "min", "max", "median", "q1", "q3"]):
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
    if "min" in features:
        x.append(np.array([np.min(combined_population, axis=0) for combined_population in combined_populations]))
    if "max" in features:
        x.append(np.array([np.max(combined_population, axis=0) for combined_population in combined_populations]))
    if "median" in features:
        x.append(np.array([np.median(combined_population, axis=0) for combined_population in combined_populations]))
    if "q1" in features:
        x.append(np.array([np.percentile(combined_population, 25, axis=0) for combined_population in combined_populations]))
    if "q3" in features:
        x.append(np.array([np.percentile(combined_population, 75, axis=0) for combined_population in combined_populations]))
    if "range" in features:
        x.append(np.array([np.max(combined_population, axis=0) - np.min(combined_population, axis=0) for combined_population in combined_populations]))
    if "interquartile_range" in features:
        x.append(np.array([np.percentile(combined_population, 75, axis=0) - np.percentile(combined_population, 25, axis=0) for combined_population in combined_populations]))

    x = np.concatenate(x, axis=1)
    return x

def bin(y, bins, verbose=False):
    y = np.where(np.array(y) < bins[0], bins[0], y)
    y = np.digitize(y, bins)
    if verbose:
        for i in range(len(bins)-1):
            print("{} samples with {} < y <= {}".format(np.sum(y == i+1), bins[i], bins[i+1]))
        print()
    return y-1

# take a list of populations and return the marginal estimated pdf of each feature
# evaluated at the query points
def get_marginal_distributions(features_list, query_points, method='kde'):    
    results = []
    n_features = len(features_list[0][0])

    for features in features_list:
        feature_distributions = []
        features = np.array(features)
        
        for i in range(n_features):
            data = features[:, i]
            kde = scipy.stats.gaussian_kde(data)
            evaluated = kde.evaluate(query_points[i])
            feature_distributions = feature_distributions + list(evaluated)

        results.append(np.array(feature_distributions))
    return np.array(results)

# takes a list of populations, finds mean and std of each feature across all populations 
# and returns n_points equidistant query points up to n_std standard deviations around the mean
def get_query_points_marginal(features_list, n_points=20, n_std=2):
    features_single_cell = np.array([feature for features in features_list for feature in features])
    mean = np.mean(features_single_cell, axis=0)
    std = np.std(features_single_cell, axis=0)
    query_points = np.transpose(np.linspace(mean - n_std*std, mean + n_std*std, n_points+2)[1:-1,:])
    return query_points

def get_fixed_size_subsample(populations, y, size=200):
    subsamples = []
    y_subsamples = []
    for x, y_ in zip(populations, y):
        n = len(x)
        if n < size:
            continue
        idx = np.random.choice(n, size, replace=False)
        subsamples.append(np.array(x)[idx, :])
        y_subsamples.append(y_)
    return np.array(subsamples).reshape(len(subsamples), -1), y_subsamples

def pickleLoader(pklFile):
    try:
        while True:
            yield pickle.load(pklFile)
    except EOFError:
        pass

# Save the features and the activation rates to a file
def save_data(file, *args):
    with open(file, 'wb') as f:
        for x in args:
            pickle.dump(x, f)

# Load the features and the activation rates from a file
def load_data(file):
    res = []
    with open(file, 'rb') as f:
        for event in pickleLoader(f):
            res.append(event)
    return tuple(res)
        
def create_dataset(control=False, k=None):
    # load data
    file = './data/bat_ifc.csv'
    samples, y_avidin, y_cd203c, y_cd63, patient_id = get_data(file, normalize_negative_control=control)

    print("Number of samples: {}".format(len(samples)))
    print("Number of patients: {}".format(len(set(patient_id))))

    # add opacity to the populations (now already done in get_data)
    # samples = add_opacity(samples)

    # train/val patients
    patient_set = list(sorted(set(patient_id)))

    if k != None:
        os.makedirs('./data/{}_fold'.format(k), exist_ok=True)
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        for i, (train_index, val_index) in enumerate(kf.split(patient_set)):
            train_val_patients = [patient_set[i] for i in train_index]
        
            # create split indices
            test_idx = [i for i in range(len(patient_id)) if not patient_id[i] in train_val_patients]
            train_idx = [i for i in range(len(patient_id)) if patient_id[i] in train_val_patients]

            # split the data
            x_train = [samples[i] for i in train_idx]
            y_train_cd63 = [y_cd63[i] for i in train_idx]
            y_train_cd203c = [y_cd203c[i] for i in train_idx]
            y_train_avidin = [y_avidin[i] for i in train_idx]

            x_test = [samples[i] for i in test_idx]
            y_test_cd63 = [y_cd63[i] for i in test_idx]
            y_test_cd203c = [y_cd203c[i] for i in test_idx]
            y_test_avidin = [y_avidin[i] for i in test_idx]

            if not control:
                save_data('./data/{}_fold/{}_train.pickle'.format(k, i), x_train, y_train_avidin, y_train_cd203c, y_train_cd63)
                save_data('./data/{}_fold/{}_test.pickle'.format(k, i), x_test, y_test_avidin, y_test_cd203c, y_test_cd63)
            else:
                save_data('./data/{}_fold_control/{}_train.pickle'.format(k, i), x_train, y_train_avidin, y_train_cd203c, y_train_cd63)
                save_data('./data/{}_fold_control/{}_test.pickle'.format(k, i), x_test, y_test_avidin, y_test_cd203c, y_test_cd63)

    else:
        train_val_patients = np.random.choice(patient_set, int(len(patient_set)*0.8), replace=False)

        # create split indices
        test_idx = [i for i in range(len(patient_id)) if not patient_id[i] in train_val_patients]
        train_idx = [i for i in range(len(patient_id)) if patient_id[i] in train_val_patients]

        # split the data
        x_train = [samples[i] for i in train_idx]
        y_train_cd63 = [y_cd63[i] for i in train_idx]
        y_train_cd203c = [y_cd203c[i] for i in train_idx]
        y_train_avidin = [y_avidin[i] for i in train_idx]

        x_test = [samples[i] for i in test_idx]
        y_test_cd63 = [y_cd63[i] for i in test_idx]
        y_test_cd203c = [y_cd203c[i] for i in test_idx]
        y_test_avidin = [y_avidin[i] for i in test_idx]

        print("Training set size: {}".format(len(x_train)))
        print("Test set size: {}".format(len(x_test)))

        if not control:
            save_data('./data/{}_populations_antiIge.pickle'.format(len(x_train)), x_train, y_train_avidin, y_train_cd203c, y_train_cd63)
            save_data('./data/{}_populations_antiIge.pickle'.format(len(x_test)), x_test, y_test_avidin, y_test_cd203c, y_test_cd63)
        else:
            save_data('./data/{}_populations_antiIge_control.pickle'.format(len(x_train)), x_train, y_train_avidin, y_train_cd203c, y_train_cd63)
            save_data('./data/{}_populations_antiIge_control.pickle'.format(len(x_test)), x_test, y_test_avidin, y_test_cd203c, y_test_cd63)

        return len(x_train), len(x_test)

def precompute_large_marginal_dataset(file_train, file_test, max_combs=2**12,
                             n_points=20, n_std=2, k=None, idx=None):
    x_train, y_train_avidin, y_train_cd203c, y_train_cd63 = load_data(file_train)
    x_test, y_test_avidin, y_test_cd203c, y_test_cd63 = load_data(file_test)

    # Shuffle the data
    xy = list(zip(x_train, y_train_avidin, y_train_cd203c, y_train_cd63))
    random.shuffle(xy)
    x_train, y_train_avidin, y_train_cd203c, y_train_cd63 = zip(*xy)
    N = len(x_train)

    print("Combining the training populations...")
    # Subsample the populations to get dataset
    np.random.seed(3)
    x_train_mixy, y_train_avidin_mixy, _, _ = subsample_populations_mixy(x_train, y_train_avidin, train_split=.99, combine_train=True, combine_test=False, max_combs=max_combs)
    np.random.seed(3)
    x_train_mixy, y_train_cd203c_mixy, _, _ = subsample_populations_mixy(x_train, y_train_cd203c, train_split=.99, combine_train=True, combine_test=False, max_combs=max_combs)
    np.random.seed(3)
    x_train_mixy, y_train_cd63_mixy, _, _ = subsample_populations_mixy(x_train, y_train_cd63, train_split=.99, combine_train=True, combine_test=False, max_combs=max_combs)
    np.random.seed(3)
    x_train_consty, y_train_avidin_consty, _, _ = subsample_populations_consty(x_train, y_train_avidin, train_split=.99, sample_size=0.75, combs_per_sample=int(max_combs/N))
    np.random.seed(3)
    x_train_consty, y_train_cd203c_consty, _, _ = subsample_populations_consty(x_train, y_train_cd203c, train_split=.99, sample_size=0.75, combs_per_sample=int(max_combs/N))
    np.random.seed(3)
    x_train_consty, y_train_cd63_consty, _, _ = subsample_populations_consty(x_train, y_train_cd63, train_split=.99, sample_size=0.75, combs_per_sample=int(max_combs/N))

    x_train = [*x_train_mixy, *x_train_consty]
    y_train_avidin = [*y_train_avidin_mixy, *y_train_avidin_consty]
    y_train_cd203c = [*y_train_cd203c_mixy, *y_train_cd203c_consty]
    y_train_cd63 = [*y_train_cd63_mixy, *y_train_cd63_consty]

    print("Estimating the marginal distributions...")
    # Get the marginal distribution features
    query_points = get_query_points_marginal(x_train, n_points=n_points, n_std=n_std)
    x_train = get_marginal_distributions(x_train, query_points)
    x_test = get_marginal_distributions(x_test, query_points)
    
    if k is None:
        save_data('./data/{}_populations_{}_combinations_precomputed_trainset_antiIge_marginal_std{}_{}.pickle'.format(N, len(x_train), n_std, n_points), x_train, y_train_avidin, y_train_cd203c, y_train_cd63)
        save_data('./data/{}_populations_precomputed_testset_antiIge_marginal_std{}_{}.pickle'.format(len(x_test), n_std, n_points), x_test, y_test_avidin, y_test_cd203c, y_test_cd63)
    else:
        save_data('./data/{}_fold/{}_train_std{}_{}.pickle'.format(k, idx, n_std, n_points), x_train, y_train_avidin, y_train_cd203c, y_train_cd63)
        save_data('./data/{}_fold/{}_test_std{}_{}.pickle'.format(k, idx, n_std, n_points), x_test, y_test_avidin, y_test_cd203c, y_test_cd63)

def precompute_large_moment_dataset(file_train, file_test, max_combs=2**12, features=["mean"], k=None, idx=None):
    x_train, y_train_avidin, y_train_cd203c, y_train_cd63 = load_data(file_train)
    x_test, y_test_avidin, y_test_cd203c, y_test_cd63 = load_data(file_test)

    # Shuffle the data
    xy = list(zip(x_train, y_train_avidin, y_train_cd203c, y_train_cd63))
    random.shuffle(xy)
    x_train, y_train_avidin, y_train_cd203c, y_train_cd63 = zip(*xy)
    N = len(x_train)

    print("Combining the training populations...")
    # Subsample the populations to get dataset
    # np.random.seed(3)
    # x_train_mixy, y_train_avidin_mixy, _, _ = subsample_populations_mixy(x_train, y_train_avidin, train_split=.99, combine_train=True, combine_test=False, max_combs=max_combs)
    # np.random.seed(3)
    # x_train_mixy, y_train_cd203c_mixy, _, _ = subsample_populations_mixy(x_train, y_train_cd203c, train_split=.99, combine_train=True, combine_test=False, max_combs=max_combs)
    # np.random.seed(3)
    # x_train_mixy, y_train_cd63_mixy, _, _ = subsample_populations_mixy(x_train, y_train_cd63, train_split=.99, combine_train=True, combine_test=False, max_combs=max_combs)
    np.random.seed(3)
    x_train_consty, y_train_avidin_consty, _, _ = subsample_populations_consty(x_train, y_train_avidin, train_split=.99, sample_size=0.8, combs_per_sample=int(max_combs/N))
    np.random.seed(3)
    x_train_consty, y_train_cd203c_consty, _, _ = subsample_populations_consty(x_train, y_train_cd203c, train_split=.99, sample_size=0.8, combs_per_sample=int(max_combs/N))
    np.random.seed(3)
    x_train_consty, y_train_cd63_consty, _, _ = subsample_populations_consty(x_train, y_train_cd63, train_split=.99, sample_size=0.8, combs_per_sample=int(max_combs/N))

    x_train = [*x_train_consty] #[*x_train_mixy, *x_train_consty]
    y_train_avidin = [*y_train_avidin_consty] #[*y_train_avidin_mixy, *y_train_avidin_consty]
    y_train_cd203c = [*y_train_cd203c_consty] #[*y_train_cd203c_mixy, *y_train_cd203c_consty]
    y_train_cd63 = [*y_train_cd63_consty] #[*y_train_cd63_mixy, *y_train_cd63_consty]

    x_train = get_statistical_moment_features(x_train, features=features)
    x_test = get_statistical_moment_features(x_test, features=features)

    # save the data 
    if k is None:
        save_data('./data/{}_populations_{}_combinations_precomputed_trainset_antiIge_{}.pickle'.format(N, len(x_train), "_".join(features)), x_train, y_train_avidin, y_train_cd203c, y_train_cd63)
        save_data('./data/{}_populations_precomputed_testset_antiIge_{}.pickle'.format(len(x_test), "_".join(features)), x_test, y_test_avidin, y_test_cd203c, y_test_cd63)    
    else:
        save_data('./data/{}_fold/{}_train_{}.pickle'.format(k, idx, "_".join(features)), x_train, y_train_avidin, y_train_cd203c, y_train_cd63)
        save_data('./data/{}_fold/{}_test_{}.pickle'.format(k, idx, "_".join(features)), x_test, y_test_avidin, y_test_cd203c, y_test_cd63)


class Single_cell_dataset(Dataset):
    def __init__(self, file, max_combs=2**10, antigen="cd63", means=None, stds=None):
        np.random.seed(3)
        random.seed(3)

        samples, y_avidin, y_cd203c, y_cd63 = load_data(file)

        if means is not None and stds is not None:
            self.means = means
            self.stds = stds
        else:
            self.means = np.mean(np.array([cell for sample in samples for cell in sample]), axis=0)
            self.stds = np.std(np.array([cell for sample in samples for cell in sample]), axis=0)

        if antigen == "cd63":
            y = y_cd63
        elif antigen == "cd203c":
            y = y_cd203c
        elif antigen == "avidin":
            y = y_avidin
        else:
            raise ValueError("Antigen not recognized.")
        
        if max_combs > 0:
            x_mixy, y_mixy, _, _ = subsample_populations_mixy(samples, y, train_split=.99, combine_train=True, combine_test=False, max_combs=max_combs)
            x_consty, y_consty, _, _ = subsample_populations_consty(samples, y, train_split=.99, sample_size=0.75, combs_per_sample=int(max_combs/len(samples)))
            self.x = [*x_mixy, *x_consty]
            self.y = [*y_mixy, *y_consty]
        else:
            self.x = samples
            self.y = y
    
    def get_normalization(self):
        return self.means, self.stds
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return ((np.array(self.x[idx]) - self.means) / self.stds).astype("float32"), self.y[idx].astype("float32")

if __name__ == "__main__":
    np.random.seed(5)
    random.seed(4)
    #train_size, test_size = create_dataset(control=False, k=5)
    # precompute_large_moment_dataset('./data/{}_populations_antiIge.pickle'.format(train_size), 
    #                         './data/{}_populations_antiIge.pickle'.format(test_size),
    #                            max_combs=2**13, features=["mean", "min", "max", "median", "std", "q1", "q3", "range", "interquartile_range"])
    # precompute_large_marginal_dataset('./data/{}_populations_antiIge.pickle'.format(train_size),
    #                                     './data/{}_populations_antiIge.pickle'.format(test_size),
    #                                     max_combs=2**13, n_points=100, n_std=4)

    k = 13
    create_dataset(k=k)
    for i in range(k):
        precompute_large_moment_dataset('./data/{}_fold/{}_train.pickle'.format(k, i), 
                                './data/{}_fold/{}_test.pickle'.format(k, i),
                                max_combs=2**12, features=["mean"], k=k, idx=i)
        # precompute_large_marginal_dataset('./data/{}_fold/{}_train.pickle'.format(k, i),
        #                                 './data/{}_fold/{}_test.pickle'.format(k, i),
        #                                 max_combs=2**12, n_points=60, n_std=3, k=k, idx=i)
