import numpy as np

#Super Class (default class)
class Predictor: 
    def __init__ (self, params={}): 
        self.weights = None
        self.params = params

    def fit(self, Xtrain, ytrain, temp):
        self.weights = np.random.rand(Xtrain.shape[1])
    
    def predict(self, Xtest):
        yhat = np.dot(Xtest, self.weights)
        return yhat
    
    def reset(self, params):
        for k in self.params: 
            if k in params: 
                self.params[k] = params[k]
    

class BatchGradientDescent(Predictor): 
    def __init__(self, params={}): 
        self.weights = None
        self.params = {'tolerance':0.0} 
        self.reset(params)

    def Errw(self,X,y):
        Xwmy = np.dot(X,self.weights) - y
        Errw = np.dot(Xwmy.T, Xwmy)
        return Errw

#Learn (Training) ** n = number of data points, m = number of features, weights = random number from features           
    def fit(self, Xtrain, ytrain):
        n,m = Xtrain.shape
        self.weights = np.random.rand(m)
        err = float('inf')
        steps = 0

        while(np.absolute(self.Errw(Xtrain,ytrain) - err) > self.params['tolerance']):
            err = self.Errw(Xtrain, ytrain)
            g = 1/float(n) * np.dot(Xtrain.T, np.dot(Xtrain, self.weights) - ytrain)
            print (g)
            self.weights = self.weights - g
            steps += 1
           # print(steps)
        print("Number of epochs: %d" % steps)

class StochasticGradientDescent(Predictor):
     def __init__(self, params={}):
        self.weights = None
        self.params = {'epoch':0}
        self.reset(params)
     #Learn (Training) ** n = number of data points, m = number of features, weights = random number from features              

     def fit(self, Xtrain, ytrain):
        n,m = Xtrain.shape
        self.weights = np.random.rand(m)
        eta0 = 1.0
        
        for i in range(self.params['epoch']):
        #print(Xtrain.shape, ytrain.shape)                                                                                                          
            ytrain = ytrain.reshape(ytrain.shape[0], 1)
            temp = np.hstack((Xtrain, ytrain))
            np.random.shuffle(temp)
            Xtrain = temp[:, :-1]
            ytrain = temp[:, -1]
            for t in range(n):
                #print(self.weights)
                g = np.dot(np.dot(Xtrain[t].T, self.weights) - ytrain[t], Xtrain[t])
               #eta = step size | 1/t
                eta = eta0 * ((t + 1) ** -1)
                self.weights = self.weights - (eta * g)


class LogisticRegression(Predictor): 
    def __init__(self, params={}):
        self.weights = None
        self.params = {'tolerance':0.0}
        self.err = []
        self.reset(params)

    def sigmoid(self, net):
        sig = 1.0/ (1 +np.exp(np.negative(net)))
        return sig

    def CrossEntropy(self, Xtrain, ytrain): 
        net = np.dot(Xtrain, self.weights)
        cross1 = np.dot(ytrain, np.log(self.sigmoid(net)))
        cross2 = np.dot((1.0 - ytrain).T, np.log(1 - self.sigmoid(net)))
        crossEnt = cross1 + cross2
        return np.negative(crossEnt)

    def fit(self, Xtrain, ytrain): 
        n,m = Xtrain.shape
        self.weights = np.random.rand(m)
        err = float('inf')
        steps = 0
        while(np.absolute(self.CrossEntropy(Xtrain,ytrain) - err) > self.params['tolerance']): #cross entropy
            err = self.CrossEntropy(Xtrain, ytrain)
            p = self.sigmoid(np.dot(Xtrain, self.weights)) #Prediction
            g = 1/float(n) * np.dot(Xtrain.T, ytrain - p)
            g = np.negative(g)
           # print (g)
            self.weights = self.weights - g
            steps += 1
            self.err.append(err)
            #print(steps)
        print("Number of epochs: %d" % steps)

    def predict(self, Xtest): 
        yhat = self.sigmoid(np.dot(Xtest, self.weights.T))
        yhat = (yhat >= 0.5) * 1
        return yhat


#SoftMax Regression, i.e., Multiclass Regression, Multinomial Logistic Regression
class SoftMaxRegression(Predictor): 
     def __init__(self, params={}):
        self.weights = None
        self.params = {'epoch':0}
        self.reset(params)
        self.K = 0
     #Learn (Training) ** n = number of data points, m = number of features, weights = random number from features
     def fit(self, Xtrain, ytrain):
        n,m = Xtrain.shape
        self.K = np.max(ytrain) + 1
        self.weights = np.random.rand(m, self.K)
        for i in range(self.params['epoch']):
            print(i)
            softM = self.SoftMax(Xtrain)
            diff = softM - self.OneHotEncoding(ytrain)
            g = np.divide(np.dot(Xtrain.T, diff), n)
            self.weights = self.weights - (g)
     
     def SoftMax(self, X):
         prob = np.exp(np.dot(X, self.weights))
         prob2 = np.divide(prob.T, np.sum(prob, axis=1))
         return prob2.T

     def OneHotEncoding(self, y):
         mat = np.zeros((len(y), self.K))
         for i, v in enumerate(y): 
             mat[i, v] = 1
         return mat
         

     def predict(self, Xtest): 
         softM = self.SoftMax(Xtest)
         yhat = softM.argmax(axis=1)
         return yhat


###TODO: Implement Softmax, One hot encoding
        #Wine loader function


