import h5py
import time
import numpy as np
import matplotlib.pyplot as plt


# Load h5py file into numpy matrices.
def loaddata():
    with h5py.File('./data/train/images_training.h5', 'r') as H:
        data_train = np.copy(H['datatrain'])
    with h5py.File('./data/train/labels_training.h5', 'r') as H:
        label_train = np.copy(H['labeltrain'])
    with h5py.File('./data/test/images_testing.h5', 'r') as T:
        data_test = np.copy(T['datatest'])
    with h5py.File('./data/test/labels_testing_2000.h5', 'r') as T:
        label_test = np.copy(T['labeltest'])

    # Number of different classes.
    labels = np.unique(label_train)
    C = len(labels)

    return data_train, label_train, data_test, label_test, C


# Calculate the euclidean distance for a single sample in the test dataset.
def distance(test, train):
    n_row = np.shape(train)[0]
    m_row = np.shape(test)[0]

    # D is a (m x n) matrix.
    # D[i, j] is the value for the distance between the number i test sample and the number j train data.
    D = np.zeros((m_row, n_row))

    for i, r_s in zip(range(m_row), test):
        for j, r_t in zip(range(n_row), train):
            D[i, j] = np.linalg.norm(r_s - r_t)
    
    return D


# Sort the points by distance.
def bubblesort(seq, label):
    flag = True
    r_seq = seq.copy()
    r_label = label.copy()

    while flag :
        flag = False
        i = 0
        while i < len(r_seq) - 1:
            temp = r_seq[i]
            temp_l = r_label[i]
            if r_seq[i] > r_seq[i+1] :
                r_seq[i] = r_seq[i+1]
                r_label[i] = r_label[i+1]

                r_seq[i+1] = temp
                r_label[i+1] = temp_l

                flag = True
            i += 1

    # Merge two list into a list of tuples.
    return list(zip(r_seq, r_label))


# Get the first K nearest neighbours and count the vote.
def vote_count(k, sorted_list, C):
    vote_count = np.zeros(C)

    i = 0
    while i <= k:
        index = sorted_list[i][1]
        vote_count[index] += 1
    
    return np.argmax(vote_count)


# Make predictions.
def predict(distance, label, k, C):
    y_pred = np.zeros(np.shape(label))

    i = 0
    for single in distance:
        sorted_list = bubblesort(single, label)
        y_pred[i] = vote_count(k, sorted_list, C)
        i += 1
    
    return y_pred


# Calculate the accuracy.
def accuracy(y_pred, y_ture):
    try:
        out = sum(y_pred == y_ture)
        ratio = out / len(y_pred)
    except TypeError as e:
        print(format(e))
        print('Please check arugments in fit method.')
    else:
        return ratio
    

if __name__ == '__main__':
    pass
