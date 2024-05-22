import data
import random
import numpy as np

file_1_train = './data/25_mixypopulations_16384_combinations_precomputed_trainset_antiIge.pickle'
file_1_test = './data/11_mixypopulations_precomputed_testset_antiIge.pickle'
file_2_train = './data/25_constypopulations_16375_combinations_precomputed_trainset_antiIge.pickle'
file_2_test = './data/11_constypopulations_precomputed_testset_antiIge.pickle'
x_train_mixy, y_train_mixy_avidin, y_train_mixy_cd203c, y_train_mixy_cd63 = data.load_data(file_1_train)
x_test, y_test_avidin, y_test_cd203c, y_test_cd63 = data.load_data(file_1_test)
x_train_consty, y_train_consty_avidin, y_train_consty_cd203c, y_train_consty_cd63 = data.load_data(file_2_train)

x_train = np.concatenate((x_train_mixy, x_train_consty), axis=0)
y_train_avidin = np.concatenate((y_train_mixy_avidin, y_train_consty_avidin), axis=0)
y_train_cd203c = np.concatenate((y_train_mixy_cd203c, y_train_consty_cd203c), axis=0)
y_train_cd63 = np.concatenate((y_train_mixy_cd63, y_train_consty_cd63), axis=0)


# shuffle the training data
np.random.seed(0)
np.random.shuffle(x_train)
np.random.seed(0)
np.random.shuffle(y_train_avidin)
np.random.seed(0)
np.random.shuffle(y_train_cd203c)
np.random.seed(0)
np.random.shuffle(y_train_cd63)

print(x_train.shape)

data.save_data('./data/25_populations_{}_combinations_precomputed_trainset_antiIge.pickle'.format(len(x_train)),x_train, y_train_avidin, y_train_cd203c, y_train_cd63 )