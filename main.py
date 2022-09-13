import numpy as np
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


# DP, only instance side-info, setting B
def get_mystery_dataset_v1(dataset_name, root_dir):
    # load the complete dataset
    Y, X1, _ = load_dataset(dataset_name, root_dir)

    triplets = [(i, j, Y[i, j]) for i in range(Y.shape[0]) for j in range(Y.shape[1])]

    train_instance_ids, test_instance_ids = train_test_split(range(Y.shape[0]), test_size=0.25, shuffle=False, random_state=42)
    
    train_triplets = [i for i in triplets if i[0] in train_instance_ids]
    test_triplets = [i for i in triplets if i[0] in test_instance_ids]
    
    return train_triplets, test_triplets, X1

# DP, only instance side-info, setting B
def get_mystery_dataset_v2(dataset_name, root_dir):
    # load the complete dataset
    Y, X1, X2 = load_dataset(dataset_name, root_dir)

    triplets = [(i, j, Y[i, j]) for i in range(Y.shape[0]) for j in range(Y.shape[1])]

    train_instance_ids, test_instance_ids = train_test_split(range(Y.shape[0]), test_size=0.25, shuffle=False, random_state=42)
    
    train_triplets = [i for i in triplets if i[0] in train_instance_ids]
    test_triplets = [i for i in triplets if i[0] in test_instance_ids]
    
    return train_triplets, test_triplets, X1, X2

# DP, only instance side-info, setting B
def get_mystery_dataset_v3(dataset_name, root_dir):
    # load the complete dataset
    Y, X1, X2 = load_dataset(dataset_name, root_dir)

    triplets = [(i, j, Y[i, j]) for i in range(Y.shape[0]) for j in range(Y.shape[1])]

    train_instance_ids, test_instance_ids = train_test_split(range(Y.shape[0]), test_size=0.25, shuffle=False, random_state=42)
    train_target_ids, test_target_ids = train_test_split(range(Y.shape[1]), test_size=0.25, shuffle=False, random_state=42)
    
    train_triplets = [i for i in triplets if i[0] in train_instance_ids and i[1] in train_target_ids]
    test_triplets = [i for i in triplets if i[0] in test_instance_ids and i[1] in test_target_ids]
    
    return train_triplets, test_triplets, X1, X2