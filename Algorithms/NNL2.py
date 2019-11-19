import numpy as np
def sigmoid(x):
    return (1/(1+np.exp(-1*x)))

def forwardPropogation(x,nh,y):
    a0 = x;
    numOfFeatures = x.shape[0]
    numOfExamples = x.shape[1];
    w1 =