import numpy as np


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
    return 2*x


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





