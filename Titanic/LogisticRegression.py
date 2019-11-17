import numpy as np


def func_sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


def func_forward_prop(w, x, y):
    yhat = func_sigmoid(np.dot(w.T, x))
    return yhat


def func_cost(y, yhat):
    m = y.shape
    cost = np.sum(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
    return cost


def func_gradient(w, alpha, y, x):
    yhat = func_forward_prop(w, x, y)
    print(yhat)
    cost = func_cost(y, yhat)
    w = w - np.dot(x, (alpha * (yhat.T - y)))
    return w, cost


def run_model(num_iter, x, y, alpha):
    costArr = np.zeros((num_iter,1))
    w = np.ones((x.shape[0], 1))
    print(w.shape)
    for i in range(0, num_iter):
        w, cost = func_gradient(w, alpha, y, x)
        costArr[i] = cost
    return w, costArr
