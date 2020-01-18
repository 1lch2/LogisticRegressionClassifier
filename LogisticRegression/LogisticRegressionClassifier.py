import h5py
import time
import numpy as np
import matplotlib.pyplot as plt

# Import h5py into numpy matrices
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

# Adding an extra column with 1 to input X martix
def preprocess(X):
    temp = np.ones((X.shape[0], 1))
    X_ = np.c_[temp, X]
    return X_


def softmax(wx):
    e_wx = np.exp(wx)
    sfm = e_wx / e_wx.sum(axis=1, keepdims=True)
    return sfm


# Calculate the gradient with regularization term,
def gredient(X, y, W, lmd=0.01):
    # Calculate the conditianal probability using softmax function.
    try:
        P = softmax(X.dot(W))  # N x C
    except ValueError as e:
        print(e)

    N = P.shape[0]
    row_n = range(N)

    # Vectorize the label array, each row has C columns.
    # a.k.a. OneHot encoding.
    Y = np.zeros((N, C))  # N x C
    Y[row_n, y] = 1

    P_Y = P - Y  # A - Y
    gre = X.T.dot(P_Y) / N  # K x C, same with W.

    R = (lmd/N) * (np.r_[np.zeros((1, W.shape[1])), W[1:]])
    return gre + R


# Cross-entropy loss function with regularization term.
def cross_entropy(X, y, W, lmd=0.01):
    P = softmax(X.dot(W))
    pred_prob = np.log(P)

    N = pred_prob.shape[0]
    row_n = range(N)

    Y = np.zeros((N, C))  # N x C
    Y[row_n, y] = 1

    loss_sum = 0
    for row_p, row_y in zip(pred_prob, Y):
        loss_sum += np.vdot(row_p, row_y)

    R = (lmd / 2*N) * np.sum(np.square(W[1:]))
    return -(loss_sum / N) + R


# Divide dataset into smaller sets, use BGD within each small sets and SGB on the whole datasets.
def fit(X, y, W, lr=0.005, iterlimit=100, batch_size=10, lmd=0.01):
    W_old = W.copy()
    itercount = 0
    N = X.shape[0]

    loss_hist = [cross_entropy(X, y, W, lmd)]  # Record the loss to make plot.
    nbatches = int(np.ceil(float(N) / batch_size))  # np.ceil，向上取整

    while itercount < iterlimit:
        itercount += 1
        mix_ids = np.random.permutation(N)  # Randomize the smaller datasets.

        # BGD
        for i in range(nbatches):
            batch_ids = mix_ids[batch_size * i : min(batch_size * (i + 1), N)]
            X_batch, y_batch = X[batch_ids], y[batch_ids]
            W -= lr * gredient(X_batch, y_batch, W, lmd)
        loss_hist.append(cross_entropy(X, y, W))
        delta = np.linalg.norm(W - W_old)
        if np.sqrt(delta) < 1e-5:
            print('Converged.\n')
            break
        W_old = W.copy()
    return W, loss_hist


# Predict the label.
def predict(W, X):
    A = softmax(X.dot(W))
    return np.argmax(A, axis=1)


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

# Tuning parameters.
def run(para):
    data_train_ = preprocess(data_train)
    data_test_ = preprocess(data_test)
    
    # Record the training time.
    time_start = time.time()
    W_init = np.random.randn(data_train_.shape[1], 10)
    W, loss_hist = fit(data_train_, label_train, W_init, batch_size=20, iterlimit=800, lr=0.005, lmd=para)
    time_end = time.time()

    label_predict = predict(W, data_test_)
    print("Accuracy of model on test set: {:.2%}".format(accuracy(label_predict, label_test[:2000])))
    print("Time: {:.3f} s.".format(time_end - time_start))
    
    acc = accuracy(label_predict, label_test[:2000])
    loss_avg = np.mean(loss_hist[-5])
    
    return loss_avg, acc

# Use the parameter list to tune lambda and learning-rate.
def tune():
    paralist = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
    losslist = []
    acclist = []

    for para in paralist:
        l, a = run(para)
        losslist.append(l)
        acclist.append(a)
    
    # Draw the line chart of lambda and loss.
    plt.plot(paralist, losslist)
    plt.xlabel('learning rate', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()
    
    # Draw the line chart of lambda and accuracy.
    plt.plot(paralist, acclist)
    plt.xlabel('learning rate', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show()


if __name__ == '__main__':
    # Preprocess the input data.
    data_train_ = preprocess(data_train)
    data_test_ = preprocess(data_test[:5000])

    # Record the training time.
    time_start = time.time()
    W_init = np.random.randn(data_train_.shape[1], 10)
    W, _losshistory = fit(data_train_, label_train, W_init, batch_size=20, iterlimit=800, lr=0.005, lmd=0.01)
    time_end = time.time()

    # Make predictions.
    label_predict = predict(W, data_test_)

    # Write prediction into file.
    h5file = h5py.File('./predicted_labels.h5', 'w')
    h5file.create_dataset('output', data=label_predict)
