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

    vdw1 = np.zeros((1000, 784))
    vdb1 = np.zeros((1000, 1))
    vdw2 = np.zeros((10, 1000))
    vdb2 = np.zeros((10, 1))

    sdw1 = np.zeros((1000, 784))
    sdb1 = np.zeros((1000, 1))
    sdw2 = np.zeros((10, 1000))
    sdb2 = np.zeros((10, 1))

    return {
        "W1": w1,
        "B1": b1,
        "W2": w2,
        "B2": b2,
        "vdW1": vdw1,
        "vdB1": vdb1,
        "vdW2": vdw2,
        "vdB2": vdb2,
        "sdW1": sdw1,
        "sdB1": sdb1,
        "sdW2": sdw2,
        "sdB2": sdb2
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
    dw2 = (1 / m) * np.dot(dz2, a1.T) + (reg_lambda/m) * w2
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

    dz1 = np.dot(w2.T, dz2) * derivative_tanh(z1)
    dw1 = (1 / m) * np.dot(dz1, x.T) + (reg_lambda/m) * w1
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    gradients = {
        "dW1": dw1,
        "dB1": db1,
        "dW2": dw2,
        "dB2": db2
    }

    return gradients


# update parameters using gradient descent
def update_parameters(params, gradients, alfa, beta1, beta2, itc):
    w1 = params['W1']
    b1 = params['B1']
    w2 = params['W2']
    b2 = params['B2']

    vdw1 = params['vdW1']
    vdb1 = params['vdB1']
    vdw2 = params['vdW2']
    vdb2 = params['vdB2']

    sdw1 = params['sdW1']
    sdb1 = params['sdB1']
    sdw2 = params['sdW2']
    sdb2 = params['sdB2']

    dw1 = gradients['dW1']
    db1 = gradients['dB1']
    dw2 = gradients['dW2']
    db2 = gradients['dB2']

    vdw1 = beta1 * vdw1 + (1 - beta1) * dw1
    vdb1 = beta1 * vdb1 + (1 - beta1) * db1
    vdw2 = beta1 * vdw2 + (1 - beta1) * dw2
    vdb2 = beta1 * vdb2 + (1 - beta1) * db2

    vdw1_c = vdw1 / (1 - beta1**itc)  # bias correction
    vdb1_c = vdb1 / (1 - beta1**itc)
    vdw2_c = vdw2 / (1 - beta1**itc)
    vdb2_c = vdb2 / (1 - beta1**itc)

    sdw1 = beta2 * sdw1 + (1 - beta2) * dw1 ** 2
    sdb1 = beta2 * sdb1 + (1 - beta2) * db1 ** 2
    sdw2 = beta2 * sdw2 + (1 - beta2) * dw2 ** 2
    sdb2 = beta2 * sdb2 + (1 - beta2) * db2 ** 2

    sdw1_c = sdw1 / (1 - beta2**itc)  # bias correction
    sdb1_c = sdb1 / (1 - beta2**itc)
    sdw2_c = sdw2 / (1 - beta2**itc)
    sdb2_c = sdb2 / (1 - beta2**itc)

    epsilon = 0.00000001

    w1 = w1 - alfa * (vdw1_c / (np.sqrt(sdw1_c) + epsilon))
    b1 = b1 - alfa * (vdb1_c / (np.sqrt(sdb1_c) + epsilon))
    w2 = w2 - alfa * (vdw2_c / (np.sqrt(sdw2_c) + epsilon))
    b2 = b2 - alfa * (vdb2_c / (np.sqrt(sdb2_c) + epsilon))

    output = {
        "W1": w1,
        "B1": b1,
        "W2": w2,
        "B2": b2,
        "vdW1": vdw1,
        "vdB1": vdb1,
        "vdW2": vdw2,
        "vdB2": vdb2,
        "sdW1": sdw1,
        "sdB1": sdb1,
        "sdW2": sdw2,
        "sdB2": sdb2
    }

    return output


def model(x, y, n_h, alfa, reg_lambda, iterations):
    n_x = x.shape[0]  # must return the number of neurons/features in input layer
    n_y = y.shape[0]  # must return the number of neurons in output layer

    cost_list = []

    params = initialize_params(n_x, n_h, n_y)

    for i in range(1, iterations):

        forward_cache = forward_propagation(x, params)

        cost = cost_function(forward_cache["A2"], y)

        gradients = back_propagation(x, y, params, forward_cache, reg_lambda)

        params = update_parameters(params, gradients, alfa, beta1=0.9, beta2=0.999, itc=i)

        cost_list.append(cost)

        if i % (iterations / 10) == 0:
            print("cost after", i, "iters is", cost)

    return params, cost_list


if __name__ == '__main__':
    X_train = np.loadtxt("data/train_X.csv", delimiter=',')
    Y_train = np.loadtxt("data/train_label.csv", delimiter=',')

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

    iterations = 1000
    n_h = 1000
    alfa = 0.002
    reg_lambda = 0.5
    parameters, cost_list = model(X_train, Y_train, n_h = n_h, alfa = alfa, reg_lambda=reg_lambda, iterations = iterations)

    print(f"Final parameter values are: {parameters}")

    # idx = int(random.randrange(0, X_test.shape[1]))
    # plt.imshow(X_test[:, idx].reshape((28, 28)), cmap='gray')
    # plt.show()

    for k in range(X_test.shape[1]):
        print("Label for test set is:")
        y_val = np.where(Y_test[:, k] == 1.)
        print(y_val[0])
        cache = forward_propagation(X_test[:, k].reshape(X_test[:, k].shape[0], 1), parameters)
        a_pred = cache['A2']
        a_pred = np.argmax(a_pred, 0)

        print("Our model says it is :", a_pred[0])