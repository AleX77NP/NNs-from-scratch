import numpy as np
import sklearn
import sklearn.datasets


def logistic(z):
    return 1. / (1+np.exp(-z))


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def hidden_activations(X, Wh, bh):
    return logistic((X @ Wh) + bh)


def output_activations(H, Wo, bo):
    return softmax((H @ Wo) + bo)


def nn(X, Wh, bh, Wo, bo):
    return output_activations(hidden_activations(X, Wh, bh), Wo, bo)


def nn_predict(X, Wh, bh, Wo, bo):
    return np.around(nn(X, Wh, bh, Wo, bo))


def loss(Y, T):
    return - (T * np.log(Y)).sum()


def error_output(Y, T):
    return Y - T


def gradient_weight_out(H, Eo):
    return H.T @ Eo


def gradient_bias_out(Eo):
    return np.sum(Eo, axis=0, keepdims=True)


def error_hidden(H, Wo, Eo):
    return np.multiply(np.multiply(H, (1-H)), (Eo @ Wo.T))


def gradient_weight_hidden(X, Eh):
    return X.T @ Eh


def gradient_bias_hidden(Eh):
    return np.sum(Eh, axis=0, keepdims=True)


def backpropagation_gradients(X, T, Wh, bh, Wo, bo):
    H = hidden_activations(X, Wh, bh)
    Y = output_activations(H, Wo, bo)

    Eo = error_output(Y, T)
    Jwo = gradient_weight_out(H, Eo)
    Jbo = gradient_bias_out(Eo)

    Eh = error_hidden(H, Wo, Eo)
    Jwh = gradient_weight_hidden(X, Eh)
    Jbh = gradient_bias_hidden(Eh)

    return [Jwh, Jbh, Jwo, Jbo]


def update_momentum(X,T, param_list, Ms, momentum_term, alfa):
    Js = backpropagation_gradients(X, T,*param_list)
    return [momentum_term * M - alfa * J for M,J in zip(Ms, Js)]


def update_params(param_list, Ms):
    return [P + M for P,M in zip(param_list, Ms)]


if __name__ == '__main__':
    # Generate the dataset
    X, t = sklearn.datasets.make_circles(
        n_samples=100, shuffle=False, factor=0.3, noise=0.1)
    T = np.zeros((100, 2))  # Define target matrix
    T[t == 1, 1] = 1
    T[t == 0, 0] = 1
    # Separate the red and blue samples for plotting
    x_red = X[t == 0]
    x_blue = X[t == 1]

    print('shape of X: {}'.format(X.shape))
    print('shape of T: {}'.format(T.shape))

    init_var = 0.1
    # Initialize hidden layer parameters
    bh = np.random.randn(1, 3) * init_var
    Wh = np.random.randn(2, 3) * init_var
    # Initialize output layer parameters
    bo = np.random.randn(1, 2) * init_var
    Wo = np.random.randn(3, 2) * init_var
    # Parameters are already initilized randomly with the gradient checking
    # Set the learning rate
    learning_rate = 0.02
    momentum_term = 0.9

    # Moments Ms = [MWh, Mbh, MWo, Mbo]
    Ms = [np.zeros_like(M) for M in [Wh, bh, Wo, bo]]

    # Start the gradient descent updates and plot the iterations
    nb_of_iterations = 300  # number of gradient descent updates
    # learning rate update rule
    lr_update = learning_rate / nb_of_iterations
    # list of loss over the iterations
    ls_loss = [loss(nn(X, Wh, bh, Wo, bo), T)]
    for i in range(nb_of_iterations):
        # Update the moments and the parameters
        Ms = update_momentum(
            X, T, [Wh, bh, Wo, bo], Ms, momentum_term, learning_rate)
        Wh, bh, Wo, bo = update_params([Wh, bh, Wo, bo], Ms)
        ls_loss.append(loss(nn(X, Wh, bh, Wo, bo), T))

    print(ls_loss)