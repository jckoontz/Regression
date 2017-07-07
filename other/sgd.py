
import numpy as np

#Cost Function
def Errw(X,y,w):
    Xwmy = np.dot(X,w) - y
    Errw = np.dot(Xwmy.T, Xwmy)
    return Errw

#Learn (Training) ** n = number of data points, m = number of features, weights = random number from features           
def fit(Xtrain, ytrain, epochs):
    n,m = Xtrain.shape
    weights = np.random.rand(m)
    eta0 = 1.0

    """while(np.absolute(Errw(Xtrain,ytrain, weights) - err) > tolerance):
        err = Errw(Xtrain, ytrain, weights)
        g = 1/float(n) * np.dot(Xtrain.T, np.dot(Xtrain, weights) - ytrain)
        weights = weights - g
        steps += 1
    print("Number of epochs: %d" % steps)"""

    for i in range(epochs):
        #print(Xtrain.shape, ytrain.shape)
        ytrain = ytrain.reshape(ytrain.shape[0], 1)
        temp = np.hstack((Xtrain, ytrain))
        np.random.shuffle(temp)
        Xtrain = temp[:, :-1]
        ytrain = temp[:, -1]
        for t in range(n):
            print(weights)
            g = np.dot(np.dot(Xtrain[t].T, weights) - ytrain[t], Xtrain[t])
            eta = eta0 * ((t + 1) ** -1)
            weights = weights - (eta * g)
    return weights

#Prediction:                                                                                                          
def predict(Xtest, weights):
    yhat = np.dot(Xtest, weights)
    return yhat

