
import numpy as np

#1. Load the data                                                                                                          
def load_csv(filename, delim):
    # TODO: 
    return np.genfromtxt(filename, delimiter=delim)

def load_diab():
    data_filename = 'datasets/X.csv'
    X = load_csv(data_filename, " ")
    target_filename = 'datasets/y.csv'
    Y = load_csv(target_filename, " ")
    return split_dataset(X,Y)

def load_susy(): 
    data_filename = 'datasets/susysubset.csv'
    DataSets = load_csv(data_filename, ",")
    X = DataSets[:, :-1]
    Y = DataSets[:, -1]
    return split_dataset(X, Y)

def split_dataset(X,y):
    s = int(X.shape[0] * 0.8)
    Xtrain = X[:s]
    Xtest = X[s:]
    ytrain = y[:s]
    ytest = y[s:]

    #2. Normalize  --- make every value between -1 and 1                                                                       
    for ii in range(Xtrain.shape[1]):
        maxval = np.max(np.abs(Xtrain[:,ii]))
        Xtrain[:,ii] = np.divide(Xtrain[:,ii], maxval)
        Xtest[:,ii] = np.divide(Xtest[:,ii], maxval)

    #3. Add Column of ones (****Add the bias*****) - MUST FOR LINEAR REGRESSION + NN                                           
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))
    return ((Xtrain, ytrain), (Xtest, ytest))

def load_iris(): 
    data_filename = 'datasets/X_iris.npy'
    X = np.load(data_filename)
    target_filename = 'datasets/y_iris.npy'
    y = np.load(target_filename)
    X = X[:, [0,3]]
    #2. Normalize  --- make every value between -1 and 1
    for ii in range(X.shape[1]):
        maxval = np.max(np.abs(X[:,ii]))
        X[:,ii] = np.divide(X[:,ii], maxval)
    return ((X, y), (X, y))

def load_wine(): 
    data_filename = 'datasets/wine_red.csv'
    DataSets = load_csv(data_filename, ";")
    X = DataSets[1:, :-1]
    Y = DataSets[1:, -1]
    Y -= min(Y)
    return split_dataset(X, Y)

