import numpy as np


def sigmoid(x):
    return (1 / (1 + np.exp(-1 * x)))


def gradient(x, nh, y, w1, w2, b1, b2):
    # x shape should be numberOfFeatures/m
    # y shape should be 1/m
    a0 = x;
    numOfFeatures = x.shape[0]
    numOfExamples = x.shape[1];
    # w1 nh x numOfFeatures
    print(w1)
    ny = y.shape[0];

    z1 = np.dot(w1, a0) + b1  # nhxm
    a1 = sigmoid(z1)  # nhxm

    z2 = np.dot(w2, a1) + b2;
    a2 = sigmoid(z2)  # a2 will be 1/m
    cost = func_cost(y, a2)
    dz2 = a2 - y
    db2 = np.sum(dz2, keepdims=True)
    dw2 = np.dot(a1, dz2.T).T  # nyxnh
    dz1 = (1 - np.square(np.tanh(z1))) * dz2  # nhx1
    dw1 = np.dot(x, dz1.T).T  # nhxnf
    db1 = np.sum(dz1, keepdims=True)
    return dw1, db1, dw2, db2, a2


def optimize(x, y, alpha, n_iter, nh, ):
    numOfFeatures = x.shape[0]
    numOfExamples = x.shape[1];
    w1 = np.random.randn(nh, numOfFeatures)
    w2 = np.random.randn(numOfExamples, nh)
    b2 = np.zeros((numOfExamples, 1))
    b1 = np.zeros((nh, 1))
    for i in range(0, n_iter):
        dw1, db1, dw2, db2, yhat = gradient(x, nh, y, w1, w2, b1, b2)
        w1 = w1 - alpha * dw1
        b1 = b1 - alpha * db1
        w2 = w2 - alpha * dw2
        b2 = b2 - alpha * db2
        cost = func_cost(y, yhat)
        print(cost)

def func_cost(y, yhat):
    y = y.reshape((y.shape[0], 1))
    yhat = yhat.reshape((yhat.shape[0], 1))
    m = y.shape[0]
    cost = (y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
    cost = np.sum(cost)
    return -cost / m


