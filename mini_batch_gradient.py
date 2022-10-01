import numpy as np
import math


def nn(x, w):
    """Output function"""
    return w * x


def loss(y, t):
    """MSE loss function"""
    return np.mean((t - y) ** 2)


def gradient(x, w, t):
    """Gradient descent"""
    return 2 * x * (nn(x, w) - t)


def delta_w(w_k, x, t, alfa=0.9):
    """Update function delta w"""
    return alfa * np.mean(gradient(w_k, x, t))  # alfa is learning rate


def f(x):  # y = f(x)
    return 2 * x


def create_mini_batches(x, y, batch_size):  # mini batches of given size
    mini_batches = []
    permutation = np.random.permutation(len(x))

    # shuffle X and Y
    x_temp = x[permutation]
    y_temp = y[permutation]
    no_mb = math.floor(x.shape[0] / batch_size)  # number of mini batches = number of row / batch_size

    for i in range(0, no_mb):
        x_mini = x_temp[i * batch_size:(i + 1) * batch_size]
        y_mini = y_temp[i * batch_size:(i + 1) * batch_size]
        mini_batches.append((x_mini, y_mini))
    return mini_batches


def mb_gradient_descent(x, y, alfa, batch_size):
    w = np.random.rand()
    error_list = []
    iters = 10
    for i in range(iters):
        mini_batches = create_mini_batches(x, y, batch_size=batch_size)
        for mini_batch in mini_batches:
            x_mini, y_mini = mini_batch
            dwm = delta_w(w, x_mini, y_mini, alfa)
            w = w - dwm
            error_list.append((w, loss(nn(x_mini, w), y_mini)))

    return w, error_list


if __name__ == '__main__':
    x = np.random.uniform(0, 1, 256)  # x input
    noise_variance = 0.2
    noise = np.random.rand(x.shape[0]) * noise_variance  # Gaussian noise error for each sample in x
    t = f(x) + noise

    w = np.random.rand()  # random value for w param

    num_of_iterations = 4
    w_loss = [(w, loss(nn(x, w), t))]

    for i in range(num_of_iterations):
        dw = delta_w(w, x, t)
        w = w - dw
        w_loss.append((w, loss(nn(x, w), t)))

    # print the final w and loss

    for i in range(0, len(w_loss)):
        print(f'w({i}): {w_loss[i][0]:.4f} \t loss: {w_loss[i][1]:.4f}')

    print("Mini-batch gradient descend")

    wm, error_list_mb = mb_gradient_descent(x, t, alfa=0.9, batch_size=32)
    print(f'W after mini batch: {wm}')
    print(f"Last w and loss: {error_list_mb[-1]}")
