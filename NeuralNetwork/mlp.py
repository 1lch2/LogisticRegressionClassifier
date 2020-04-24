import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import animation


class Activation(object):
    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_deriv(self, a):
        # a = np.tanh(x)
        return 1.0 - a ** 2

    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_deriv(self, a):
        # a = logistic(x)
        return self.__logistic(a) * (1 - self.__logistic(a))

    def __relu(self, x):
        return x * (x > 0)

    def __relu_deriv(self, a):
        return 1 * (a > 0)

    def __softmax(self, x):
        x -= np.max(x)  # Avoid overflow
        e_wx = np.exp(x)
        return e_wx / e_wx.sum(axis=1, keepdims=True)

    def __softmax_deriv(self, a):
        return a * (1 - a)

    def __init__(self, activation='tanh'):
        """
        self.f:         activation function
        self.f_deriv:   derivative of activation
        """
        if activation == 'logistic':
            self.f = self.__logistic
            self.f_deriv = self.__logistic_deriv
        elif activation == 'tanh':
            self.f = self.__tanh
            self.f_deriv = self.__tanh_deriv
        elif activation == 'relu':
            self.f = self.__relu
            self.f_deriv = self.__relu_deriv
        elif activation == 'softmax':
            self.f = self.__softmax
            self.f_deriv = self.__softmax_deriv


class HiddenLayer(object):
    def __init__(self, n_in, n_out,
                 activation_last_layer='tanh', activation='tanh', dropout=None, W=None, b=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: string
        :param activation: Non linearity to be applied in the hidden
                           layer

        :type dropout: int
        :param dropout: dropout rate for each layer
        """
        self.input = None
        self.activation = Activation(activation).f

        # activation deriv of last layer
        self.activation_deriv = None
        if activation_last_layer:
            self.activation_deriv = Activation(activation_last_layer).f_deriv

        # Xavier Initialization
        self.W = np.random.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)
        )
        if activation == 'logistic':
            self.W *= 4

        self.b = np.zeros(n_out, )

        # DropOut parameters
        self.dropout = dropout
        self.drop_list = None

        # Momenmtum term for W and b.
        self.vw = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    def forward(self, input, train_mode):
        '''
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)
        '''
        lin_output = np.dot(input, self.W) + self.b
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )

        if train_mode:
            # Inverted DropOut
            if self.dropout:
                self.drop_list = np.random.binomial(1, (1 - self.dropout), self.output.shape[-1]) / (1 - self.dropout)
                self.output = self.output * self.drop_list

        self.input = input
        return self.output

    def backward(self, delta, output_layer=False):
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grad_b = delta

        # Inverted DropOut
        if self.dropout:
            self.grad_W =  self.drop_list * self.grad_W

        if self.activation_deriv:
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input)
        return delta


class MLP:
    """
    """

    def __init__(self, layers, activation=[None, 'tanh', 'tanh'], dropout=None):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        # initialize layers
        self.layers = []
        self.params = []
        self.dropout = dropout
        self.train_mode = True
        self.activation = activation

        for i in range(len(layers) - 1):
            self.layers.append(HiddenLayer(layers[i], layers[i + 1], activation[i], activation[i + 1], self.dropout[i]))


    def forward(self, input):
        for layer in self.layers:
            output = layer.forward(input, self.train_mode)
            input = output

        return output

    def cross_entropy_loss(self, y_ture, y_pred, alpha):
        activation_deriv = Activation(self.activation[-1]).f_deriv

        y_pred[y_pred == 0] = 1e-8  # Avoid -inf result

        loss = - np.sum(y_ture * np.log(y_pred))
        delta = y_pred - y_ture

        return loss, delta

    # Abandoned
    def criterion_MSE(self, y, y_hat):
        activation_deriv = Activation(self.activation[-1]).f_deriv
        # MSE
        error = y - y_hat
        loss = 0.5 * np.sum(error ** 2)

        # calculate the delta of the output layer
        delta = -error * activation_deriv(y_hat)

        return loss, delta

    def backward(self, delta):
        delta = self.layers[-1].backward(delta, output_layer=True)
        for layer in reversed(self.layers[:-1]):
            delta = layer.backward(delta)

    # SGD with Momentum
    def update(self, lr, gamma, alpha):
        for layer in self.layers:
            layer.vw = gamma * layer.vw + lr * layer.grad_W
            layer.W = (1 - lr * alpha) * layer.W - layer.vw # Weight decay term for W.

            layer.vb = gamma * layer.vb + lr * layer.grad_b
            layer.b = layer.b - layer.vb

    def fit(self, X, y, X_val, y_val, learning_rate=0.01, gamma=0.9, alpha=0, epochs=100, batch_size=30):
        """
        Online learning.
        :param X: Input data or features
        :param y: Input targets
        :param learning_rate: parameters defining the speed of learning
        :param gamma: parameter for momentum
        :param alpha: parameter for weight-decay
        :param epochs: number of times the dataset is presented to the network for learning
        :param batch_size: number of training samples in each batch
        """
        X = np.array(X)
        y = np.array(y)

        # One-hot encoding
        N = np.shape(X)[0]
        row_n = range(N)
        Y = np.zeros((N, 10))  # n_sample x n_class
        Y[row_n, y] = 1


        to_return = np.zeros(epochs)
        acc = np.zeros(epochs)
        _acc = np.zeros(epochs)
        nbatches = int(np.ceil(float(N) / batch_size))  # Number of batches
        loss = np.zeros(nbatches)

        # Mini-batch training
        for k in range(epochs):
            mix_ids = np.random.permutation(N)  # Randomize the smaller datasets.
            for i in range(nbatches):
                batch_ids = mix_ids[batch_size * i: min(batch_size * (i + 1), N)]
                X_batch, y_batch = X[batch_ids], Y[batch_ids]

                # forward pass
                self.train_mode = True
                y_pred = self.forward(X_batch)

                # backward pass
                loss[i], delta = self.cross_entropy_loss(y_batch, y_pred, alpha)
                self.backward(delta)

                # update
                self.update(learning_rate, gamma, alpha)

            to_return[k] = np.mean(loss)

            self.predict(X_val)
            acc[k] = self.accuracy(y_val)

            self.predict(X[10000:31000])
            _acc[k] = self.accuracy(y[10000:31000])
            print("[epoch " + str(k) + "] loss: " + str(to_return[k]) + ", test accuracy: {:.2%}".format(acc[k]) + ", train accuracy: {:.2%}".format(_acc[k]))

        return to_return, acc, _acc

    def predict(self, x, batch_size=30):
        X = np.array(x)
        N = X.shape[0]
        output = np.zeros((N, 10))

        for i in range(0, N, batch_size):
            X_batch = X[i: i + batch_size]

            self.train_mode = False
            y_pred = self.forward(X_batch)
            output[i: i + batch_size] = y_pred.copy()

        out = np.argmax(output, axis=1)
        self.label_predict = out
        # return output

    def accuracy(self, label_validation):
        out = sum(self.label_predict == label_validation)
        return out / len(self.label_predict)


# TODO: Batch Normalization


# Load dataset from h5 files.
def load_data():
    with h5py.File('./data/train_128.h5', 'r') as H:
        data_train = np.copy(H['data'])
    with h5py.File('./data/train_label.h5', 'r') as H:
        label_train = np.copy(H['label'])
    with h5py.File('./data/test_128.h5', 'r') as T:
        data_test = np.copy(T['data'])

    return data_train, label_train, data_test


def normalization(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def whitening(x):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    return (x - x_mean) / x_std


def main():
    # Load data
    data_train, label_train, data_test = load_data()
    N_feature = np.shape(data_train)[1]  # Number of feature

    # Whitening
    data_train = whitening(data_train)

    # Validation set
    data_val, label_val = data_train[-12000:], label_train[-12000:]

    # Initialize an MLP object.
    nn = MLP([128, 256, 64, 10], [None, 'relu', 'relu', 'softmax'], [0.4, 0.4, 0, 0])
    loss_list, acc_list, _acc_list= nn.fit(data_train[:48000], label_train[:48000], data_val, label_val, learning_rate=0.0001,
                                 epochs=30)

    # # Draw chart
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    #
    # ax1.plot(range(len(loss_list)), loss_list, 'r')
    # ax1.set_ylabel("loss", fontsize=12)
    #
    # ax2 = ax1.twinx()
    # ax2.plot(range(len(acc_list)), acc_list, 'b')
    # ax2.set_xlabel("epoch", fontsize=12)
    # ax2.set_ylabel("accuracy", fontsize=12)
    # plt.show()

    plt.plot(_acc_list, label='train')
    plt.plot(acc_list, label='val')
    plt.tight_layout()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
