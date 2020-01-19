import h5py
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


# Load h5py file into numpy matrices.
def loaddata():
    with h5py.File('../data/train/images_training.h5', 'r') as H:
        data_train = np.copy(H['datatrain'])
    with h5py.File('../data/train/labels_training.h5', 'r') as H:
        label_train = np.copy(H['labeltrain'])
    with h5py.File('../data/test/images_testing.h5', 'r') as T:
        data_test = np.copy(T['datatest'])
    with h5py.File('../data/test/labels_testing_2000.h5', 'r') as T:
        label_test = np.copy(T['labeltest'])

    # Number of different classes.
    labels = np.unique(label_train)
    C = len(labels)

    print('Dataset loaded.')
    return data_train, label_train, data_test, label_test, C


# Calculate the euclidean distance for a single sample in the test dataset.
def distance(test, train):
    n_row = np.shape(train)[0]
    m_row = np.shape(test)[0]

    # D is a (m x n) matrix.
    # D[i, j] is the value for the distance between the number i test sample and the number j train data.

    start_ = time.time()    
    t = datetime.now()
    print('L2 calculation start: ' + str(t.strftime("%Y.%m.%d %H:%M:%S")))

    D = np.zeros((m_row, n_row))
    for i, r_s in zip(range(m_row), test):
        for j, r_t in zip(range(n_row), train):
            D[i, j] = np.linalg.norm(r_s - r_t)
    
    end_ = time.time()

    print('L2 distance calculation completed.')
    print('Time cost: ' + str(end_ - start_) + ' s.')
    print(np.shape(D))
    return D


# Sort the points by distance.
# Bubble sort algorithm.
# Time cost too high to use.
def bubblesort(seq, label):
    flag = True
    r_seq = seq.copy()
    r_label = label.copy()

    while flag:
        flag = False
        i = 0
        while i < len(r_seq) - 1:
            temp = r_seq[i]
            temp_l = r_label[i]
            if r_seq[i] > r_seq[i+1]:
                r_seq[i] = r_seq[i+1]
                r_label[i] = r_label[i+1]
                # FIXME: index out of bound?

                r_seq[i+1] = temp
                r_label[i+1] = temp_l

                flag = True
            i += 1

    # Merge two list into a list of tuples.
    return list(zip(r_seq, r_label))


# Quick sort.
# TODO: Need modification.
def quicksort(seq, low, high):
    i = low
    j = high

    if low < high:
        base = seq[low]
        while i < j:
            while seq[j] > base and j > i:
                j -= 1
            if j > i:
                seq[i] = seq[j]
                i += 1
            
            while seq[i] < base and i < j:
                i += 1
            if i < j:
                seq[j] = seq[i]
                j -= 1
            
        seq[i] = base

        quicksort(seq, low, i-1)
        quicksort(seq, i+1, high)


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

    t = datetime.now()
    print('Counting vote: ' + str(t.strftime("%Y.%m.%d %H:%M:%S")))

    # Traverse each row in the distance matrix.
    i = 0
    for single in distance:
        sorted_list = bubblesort(single, label) # TODO: Transfer buble sort to quick sort.
        y_pred[i] = vote_count(k, sorted_list, C)
        i += 1
    return y_pred


# Calculate the accuracy.
def accuracy(y_pred, y_true):
    out = sum(y_pred == y_true)
    return out / len(y_pred)


# Classifier runs from here.
def main(k=3):
    t = datetime.now()
    print('Training start: ' + str(t.strftime("%Y.%m.%d %H:%M:%S")))

    data_train, label_train, data_test, label_test, C = loaddata()
    
    time_start = time.time()
    
    # Use a small portion of data for test.
    D = distance(data_test[:500], data_train[:5000])
    label_predict = predict(D, label_train[:5000], k, C)
    time_end = time.time()

    acc = accuracy(label_predict, label_test[:500])
    print("Accuracy of model on test set: {:.2%}".format(acc))
    print("Time: {:.3f} s.".format(time_end - time_start))

    # TODO: Remove annotation after testing.
    # # Export the prediction into h5py file.
    # h5file = h5py.File('./predicted_labels.h5', 'w')
    # h5file.create_dataset('output', data=label_predict)


# Tuning parameter K and draw a plot.
def tuning():
    k_list = range(10)

    acc_list = []
    for _k in k_list:
        data_train, label_train, data_test, label_test, C = loaddata()

        print('Tuning for K = ' + str(_k))
        D = distance(data_test, data_train)
        label_predict = predict(D, label_train, _k, C)
        acc = accuracy(label_predict, label_test[:2000])

        acc_list.append(acc)

    # Draw the line chart of K and accuracy.
    plt.plot(k_list, acc_list)
    plt.xlabel('K', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()


if __name__ == '__main__':
    main()

    # TODO: 10-fold cross-validation.
    # TODO: Jupyter notebook format.
