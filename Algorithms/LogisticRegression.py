import numpy as np


def func_sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


def func_forward_prop(w, x, y):
    yhat = func_sigmoid(np.dot(w.T, x))
    return yhat


def func_cost(y, yhat):
    m = y.shape
    cost = y * np.log(yhat) + (1 - y) * np.log(1 - yhat)
    return cost


def func_gradient(w, alpha, y, x):
    yhat = func_forward_prop(w, x, y)
    w = w - np.dot(x, (alpha * (yhat - y)))
    return w


def run_model(num_iter, x, y, alpha):
    w = np.zeros(x.shape[1])
    for i in range(0, num_iter):
        w = func_gradient(w, alpha, y, x)
