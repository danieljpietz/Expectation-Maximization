import numpy as np
from mnist import MNIST

X_train = None
X_test = None
labels_train = None
labels_test = None
d = None
n_train = None
n_test = None
labels_train_mat = None
labels_test_mat = None
n_train = None
n_test = None
labels_binary_train = None
labels_binary_test = None

# Load the data from the local MNIST path
def load_dataset():
    global X_train, X_test, labels_train, labels_test, d, n_train, n_test
    mndata = MNIST('MNIST')
    d = 784
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train / 255.0
    X_test = X_test / 255.0


def one_hot_encode(classes=None, garbage=False):
    global labels_train_mat, labels_test_mat, X_train, X_test, labels_train, labels_test, d

    if classes is None:
        labels_train_mat = np.zeros((np.size(labels_train), 10))
        labels_test_mat = np.zeros((np.size(labels_test), 10))
        for i in range(0, np.size(labels_train)):
            labels_train_mat[i][labels_train[i]] = 1
        for i in range(0, np.size(labels_test)):
            labels_test_mat[i][labels_test[i]] = 1
    else:
        train_bad_inds = []
        test_bad_inds = []
        if garbage:
            listLen = len(classes) + 1
        else:
            listLen = len(classes)
        labels_train_mat = np.zeros((0, listLen))
        labels_test_mat = np.zeros((0, listLen))
        for i in range(0, np.size(labels_train)):
            num = labels_train[i]
            label_temp = [0 for i in range(listLen)]
            if num in classes:
                label_temp[classes.index(num)] = 1
            elif garbage:
                label_temp[-1] = 1
            else:
                train_bad_inds.append(i)
            labels_train_mat = np.vstack([labels_train_mat, label_temp])
        X_train = np.delete(X_train, train_bad_inds, axis=0)
        for i in range(0, np.size(labels_test)):
            num = labels_test[i]
            label_temp = [0 for i in range(listLen)]
            if num in classes:
                label_temp[classes.index(num)] = 1
            elif garbage:
                label_temp[-1] = 1
            else:
                test_bad_inds.append(i)
            labels_test_mat = np.vstack([labels_test_mat, label_temp])
        X_test = np.delete(X_test, test_bad_inds, axis=0)


def binary_encode(classes):
    global labels_binary_train, labels_binary_test, X_train, X_test
    labels_binary_train = np.zeros((np.size(labels_train), 1))
    labels_binary_test = np.zeros((np.size(labels_test), 1))
    labels_binary_train[labels_train == classes[0]] = 1
    labels_binary_train[labels_train == classes[1]] = -1
    X_train = X_train[tuple((labels_binary_train != 0).transpose().tolist())]
    labels_binary_train = labels_binary_train[labels_binary_train != 0]

    labels_binary_test[labels_test == classes[0]] = 1
    labels_binary_test[labels_test == classes[1]] = -1
    X_test = X_test[tuple((labels_binary_test != 0).transpose().tolist())]
    labels_binary_test = labels_binary_test[labels_binary_test != 0]
    labels_binary_train = np.reshape(labels_binary_train, (np.size(labels_binary_train), 1))

    labels_binary_train = np.reshape(labels_binary_train, (np.size(labels_binary_train), 1))
    labels_binary_test = np.reshape(labels_binary_test, (np.size(labels_binary_test), 1))


def load_and_encode(classes=None):
    global n_train, n_test
    load_dataset()
    binary_encode(classes)
    n_train = np.size(X_train, 1)
    n_test = np.size(X_test, 1)
