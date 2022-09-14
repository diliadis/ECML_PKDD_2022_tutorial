import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(dataset_name, root_dir):
    if (dataset_name == 'ern') or (dataset_name == 'srn'):
        Y = np.genfromtxt(root_dir+'/'+dataset_name+'/'+'Y.txt', delimiter=',')
        X1 = np.genfromtxt(root_dir+'/'+dataset_name+'/'+'X1.txt', delimiter=',')
        X2 = np.genfromtxt(root_dir+'/'+dataset_name+'/'+'X2.txt', delimiter=',')
    else:
        Y = np.genfromtxt(root_dir+'/'+dataset_name+'/'+'Y.txt')
        X1 = np.genfromtxt(root_dir+'/'+dataset_name+'/'+'X1.txt')
        X2 = np.genfromtxt(root_dir+'/'+dataset_name+'/'+'X2.txt')

    return Y, X1, X2


def get_mystery_dataset(dataset_name, root_dir, mode=0):
    valid_modes = [0, 1, 2]
    if mode not in valid_modes:
        raise AttributeError('Please use one of the valid modes: '+str(valid_modes))

    # load the complete dataset
    Y, X1, X2 = load_dataset(dataset_name, root_dir)

    triplets = [(i, j, Y[i, j]) for i in range(Y.shape[0]) for j in range(Y.shape[1])]
    if mode in [0, 1]:
        Y_train, Y_test, X1_train, X1_test = train_test_split(Y, X1, test_size=0.25, shuffle=False, random_state=42)
    else:
        train_instance_ids, test_instance_ids = train_test_split(range(Y.shape[0]), test_size=0.25, shuffle=False, random_state=42)
        train_target_ids, test_target_ids = train_test_split(range(Y.shape[1]), test_size=0.25, shuffle=False, random_state=42)    

        Y_train, Y_test = Y[train_instance_ids, :][:, train_target_ids], Y[test_instance_ids, :][:, test_target_ids]
        X1_train, X1_test = X1[train_instance_ids, :], X1[test_instance_ids, :]
        X2_train, X2_test = X2[train_target_ids, :], X2[test_target_ids, :]

    if mode == 0:
        return Y_train, Y_test, X1_train, X1_test
    elif mode == 1:
        return Y_train, Y_test, X1_train, X1_test, X2
    else:
        return Y_train, Y_test, X1_train, X1_test, X2_train, X2_test