import numpy as np
import matplotlib.pyplot as plt
import random


# Activation functions
def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    expX = np.exp(x)
    return expX / np.sum(expX, axis=0)


# Activation functions derivates
def derivative_tanh(x):
    return 1 - np.power(np.tanh(x), 2)


def derivative_relu(x):
    return np.array(x > 0, dtype=np.float32)


# random values at the start for params
def initialize_params(n_x, n_h, n_y):
    w1 = np.random.randn(n_h, n_x) * 0.001
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.001
    b2 = np.zeros((n_y, 1))

    return {
        "W1": w1,
        "B1": b1,
        "W2": w2,
        "B2": b2
    }


# forward propagation
def forward_propagation(x, params):
    w1 = params["W1"]
    b1 = params["B1"]
    w2 = params["W2"]
    b2 = params["B2"]

    z1 = np.dot(w1, x) + b1
    a1 = tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = softmax(z2)

    output = {
        "Z1": z1,
        "A1": a1,
        "Z2": z2,
        "A2": a2
    }

    return output


# cost function
def cost_function(a2, y):
    m = y.shape[1]
    cost = -(1 / m) * np.sum(y * np.log(a2))

    return cost


# backpropagation
def back_propagation(x, y, params, forward_cache, reg_lambda):
    m = x.shape[1]

    w1 = params['W1']
    b1 = params['B1']
    w2 = params['W2']
    b2 = params['B2']

    z1 = forward_cache["Z1"]
    a1 = forward_cache["A1"]
    z2 = forward_cache["Z2"]
    a2 = forward_cache["A2"]

    dz2 = a2 - y
    dw2 = (1 / m) * np.dot(dz2, a1.T) + (reg_lambda / m) * w2  # regularization
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

    dz1 = np.dot(w2.T, dz2) * derivative_tanh(z1)
    dw1 = (1 / m) * np.dot(dz1, x.T) + (reg_lambda / m) * w1
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    gradients = {
        "dW1": dw1,
        "dB1": db1,
        "dW2": dw2,
        "dB2": db2
    }

    return gradients


# update parameters using gradient descent
def update_parameters(params, gradients, alfa):
    w1 = params['W1']
    b1 = params['B1']
    w2 = params['W2']
    b2 = params['B2']

    dw1 = gradients['dW1']
    db1 = gradients['dB1']
    dw2 = gradients['dW2']
    db2 = gradients['dB2']

    w1 = w1 - alfa * dw1
    b1 = b1 - alfa * db1
    w2 = w2 - alfa * dw2
    b2 = b2 - alfa * db2

    output = {
        "W1": w1,
        "B1": b1,
        "W2": w2,
        "B2": b2
    }

    return output


def model(x, y, alfa, reg_lambda, iterations, params):
    cost_list = []

    for i in range(iterations):

        forward_cache = forward_propagation(x, params)

        cost = cost_function(forward_cache["A2"], y)

        gradients = back_propagation(x, y, params, forward_cache, reg_lambda)

        params = update_parameters(params, gradients, alfa)

        cost_list.append(cost)

        if i % (iterations / 10) == 0:
            print("cost after", i, "iters is", cost)

    return params, cost_list


if __name__ == '__main__':
    X_train = np.loadtxt("data/train_X.csv", delimiter=',')
    Y_train = np.loadtxt("data/train_label.csv", delimiter=',')

    # 10 folds, 100 rows per fold
    folds = np.array_split(X_train, 10)
    folds_labels = np.array_split(Y_train, 10)

    X_train = X_train.T
    Y_train = Y_train.T

    X_test = np.loadtxt("data/test_X.csv", delimiter=',')
    Y_test = np.loadtxt("data/test_label.csv", delimiter=',')

    X_test = X_test.T
    Y_test = Y_test.T

    print("shape of X_train : ", X_train.shape)
    print("shape of Y_train : ", Y_train.shape)

    print("shape of X_test : ", X_test.shape)
    print("shape of Y_test : ", Y_test.shape)

    # index = int(random.randrange(0, X_train.shape[1]))
    # plt.imshow(X_train[:, index].reshape((28, 28)), cmap='gray')
    # plt.show()

    iterations = 100
    n_x = X_train.shape[0]
    n_h = 1000
    n_y = Y_train.shape[0]
    alfa = 0.02

    parameters = initialize_params(n_x, n_h, n_y)

    for i in range(10):
        folds_trains = folds.copy()
        fold_test = folds[i].T

        folds_trains_labels = folds_labels.copy()
        fold_test_labels = folds_labels[i].T

        # separate training and test data for this fold
        folds_trains = np.concatenate(folds_trains[:i] + folds_trains[i + 1:]).T
        folds_trains_labels = np.concatenate(folds_trains_labels[:i] + folds_trains_labels[i + 1:]).T

        reg_lambda = 0.1 * i + 0.1

        print(f"Lambda is {reg_lambda} for cross validation {i}")

        parameters, cost_list = model(folds_trains, folds_trains_labels, alfa=alfa, reg_lambda=reg_lambda,
                                      iterations=iterations, params=parameters)

        correct = 0
        for k in range(fold_test.shape[1]):
            print("Label for test set is:")
            y_val = np.where(fold_test_labels[:, k] == 1.)
            print(y_val[0][0])
            cache = forward_propagation(fold_test[:, k].reshape(fold_test[:, k].shape[0], 1), parameters)
            a_pred = cache['A2']
            a_pred = np.argmax(a_pred, 0)

            if y_val[0][0] == a_pred:
                correct = correct + 1

            print("Our model says it is :", a_pred[0])
        print(f"Correct answers percentage: {correct / fold_test.shape[1]}")

    idx = int(random.randrange(0, X_test.shape[1]))
    plt.imshow(X_test[:, idx].reshape((28, 28)), cmap='gray')
    plt.show()

    cache_o = forward_propagation(X_test[:, idx].reshape(X_test[:, idx].shape[0], 1), parameters)
    a_pred_o = cache_o['A2']
    a_pred_o = np.argmax(a_pred_o, 0)

    print("Our model says it is :", a_pred_o[0])
