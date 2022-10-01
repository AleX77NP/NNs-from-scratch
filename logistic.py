import numpy as np
import matplotlib
import matplotlib.pyplot as plt  # Plotting library
from matplotlib import cm # Colormaps
from matplotlib.colors import colorConverter, ListedColormap


def logistic(z):
    return 1. / (1+np.exp(-z))


def nn(x, w):
    return logistic(x.dot(w.T))


def nn_predict(x, w):
    return np.around(nn(x, w))


def loss(y, t):
    return np.mean(np.multiply(t, np.log(y)) + np.multiply(np.log(1-t), np.log(1-y)))


def gradient(w, x, t):
    return (nn(x, w) - t).T * x


def delta_w(w_k, x, t, alfa):
    return alfa * gradient(w_k, x, t)


if __name__ == '__main__':
    samples_num_per_class = 20
    red_mean = (-1., 0)
    blue_mean = (1., 0)

    x_red = np.random.rand(samples_num_per_class, 2) + red_mean
    x_blue = np.random.rand(samples_num_per_class, 2) + blue_mean

    X = np.vstack((x_red, x_blue))
    t = np.vstack((np.zeros((samples_num_per_class, 1)),
                  np.ones((samples_num_per_class, 1))))

    # plt.figure(figsize=(6, 4))
    # plt.plot(x_red[:, 0], x_red[:, 1], 'r*', label='class: red star')
    # plt.plot(x_blue[:, 0], x_blue[:, 1], 'bo', label='class: blue circle')
    # plt.legend(loc=2)
    # plt.xlabel('$x_1$', fontsize=12)
    # plt.ylabel('$x_2$', fontsize=12)
    # plt.axis([-3, 4, -4, 4])
    # plt.title('red star vs. blue circle classes in the input space')
    # plt.show()

    w = np.asmatrix([-4, -2])
    alfa = 0.05

    iterations = 20
    w_iter = [w]

    for i in range(iterations):
        dw = delta_w(w, X, t, alfa)
        w = w - dw
        w_iter.append(w)

    # weights values
    for i in range(0, len(w_iter)):
        print(f'w({i}): {w_iter[i]}')
