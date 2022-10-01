import numpy as np


def rbf(zh):
    return np.exp(-zh ** 2)


def logistic(zo):
    return 1. / (1 + np.exp(-zo))


def hidden_activations(x, wh):
    return rbf(x * wh)


def output_activations(h, bo):
    return logistic(h + bo)


def nn(x, wh, bo):
    return output_activations(hidden_activations(x, wh), bo)


def nn_predict(x, wh, bo):
    return np.around(nn(x, wh, bo))


def loss(y, t):
    return -np.mean(
        (t * np.log(y)) + ((1 - t) * np.log(1 - y))
    )


def loss_for_param(x, wh, bo, t):
    return loss(nn(x, wh, bo), t)


def gradient_output(y, t):
    return y - t


def gradient_bias_output(grad_output):
    return grad_output


def gradient_hidden(grad_output):
    return grad_output


def gradient_weight_hidden(x, zh, h, grad_hidden):
    return - x * (2 * zh * h) * grad_hidden


def backpropagation_update(x, t, wh, bo, alfa):
    zh = x * wh
    h = rbf(zh)
    y = output_activations(h, bo)

    grad_output = gradient_output(y, t)

    d_bo = alfa * gradient_bias_output(grad_output)
    grad_hidden = gradient_hidden(grad_output)
    d_wh = alfa * gradient_weight_hidden(x, zh, h, grad_hidden)

    return float(np.mean(wh - d_wh)), float(np.mean(bo - d_bo))


if __name__ == '__main__':
    nb_of_samples_per_class = 20
    blue_mean = 0
    red_left_mean = -2
    red_right_mean = 2

    std_dev = 0.5
    xs_blue = np.random.randn(
        nb_of_samples_per_class, 1) * std_dev + blue_mean
    xs_red = np.vstack((
        np.random.randn(
            nb_of_samples_per_class // 2, 1) * std_dev + red_left_mean,
        np.random.randn(
            nb_of_samples_per_class // 2, 1) * std_dev + red_right_mean
    ))

    x = np.vstack((xs_blue, xs_red))
    t = np.vstack((np.ones((xs_blue.shape[0], 1)),
                   np.zeros((xs_red.shape[0], 1))))
    wh = 2.3
    bo = 1.4

    alfa = 2.0
    iterations = 20

    params_loss = [(wh, bo, loss_for_param(x, wh, bo, t))]

    for i in range(iterations):
        wh, bo = backpropagation_update(x, t, wh, bo, alfa)
        params_loss.append((wh, bo, loss_for_param(x, wh, bo, t)))

    final_loss = loss_for_param(x, wh, bo, t)
    print(f'Final loss is {final_loss:.2f} for weights wh: {wh} and bo: {bo}')