from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

""" Splits data into training and testing set preserving the ratio between
class categories """


def split_data(x_data, y_data, train_ratio):
    sss = StratifiedShuffleSplit(y_data, 1, test_size=1 - train_ratio, random_state=0)
    for train_index, test_index in sss:
        x_train, x_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
    train = {"data": x_train, "labels": y_train}
    test = {"data": x_test, "labels": y_test}
    return train, test


"""Centers mean at zero and standard deviation to 1 using training data"""


def preprocess_data(train, test):
    scaler = StandardScaler()
    scaler.fit(train['data'])
    train_x = scaler.transform(train['data'])
    # Apply same transformation to test data using training data
    test_x = scaler.transform(test['data'])
    # Update new data
    new_train = train
    new_train['data'] = train_x
    new_test = test
    new_test['data'] = test_x
    return new_train, new_test


""" Computes the PCA of the data given the input variance to support"""


def compute_PCA(train, test, var_ratio):
    pca = PCA()
    pca.fit(train['data'])
    # Get number of components so variance is above given threshold
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    num_components = min(np.where(cumulative_var > var_ratio)[0])
    # Transform data
    new_train = pca.transform(train)[:][range(1, num_components + 1)]
    new_test = pca.transform(test)[range(1, num_components + 1)]
    # Update new data
    ntrain = train
    new_train['data'] = new_train
    ntest = test
    new_test['data'] = new_test
    return ntrain, ntest


"""" Computes stratified K fold on the data """""


def compute_kfolds(labels, data, k):
    skf = StratifiedKFold(labels, n_folds=k)
    folds = []
    for i, train_index, test_index in enumerate(skf):
        folds[i]['train']['data'] = data[train_index]
        folds[i]['train']['labels'] = labels[train_index]
        folds[i]['test']['data'] = data[test_index]
        folds[i]['test']['labels'] = labels[test_index]
    return folds
