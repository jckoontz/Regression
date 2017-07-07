
import numpy as np

#Cost Function
def Errw(X,y,w):
    Xwmy = np.dot(X,w) - y
    Errw = np.dot(Xwmy.T, Xwmy)
    return Errw

#Learn (Training) ** n = number of data points, m = number of features, weights = random number from features           
def fit(Xtrain, ytrain, tolerance):
    n,m = Xtrain.shape
    weights = np.random.rand(m)
    err = float('inf')
    steps = 0

    while(np.absolute(Errw(Xtrain,ytrain, weights) - err) > tolerance):
        err = Errw(Xtrain, ytrain, weights)
        g = 1/float(n) * np.dot(Xtrain.T, np.dot(Xtrain, weights) - ytrain)
        weights = weights - g
        steps += 1
    print("Number of epochs: %d" % steps)
    return weights

#Prediction:                                                                                                          
def predict(Xtest, weights):
    yhat = np.dot(Xtest, weights)
    return yhat

