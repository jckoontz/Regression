

from sklearn.metrics import accuracy_score
import numpy as np
import sys
import argparse 
import data_loader as dl
import predict_algs as pa
import sgd
import matplotlib.pyplot as plt

"""TODO: 
Add step size argument
Add to script step sizes 0-1
"""

def parse_arguments(): 
    parser = argparse.ArgumentParser(description="Gradient Descent")
    parser.add_argument('-t', '--tolerance', type=float, default=1.0, help="Tolerance level for Batch")
    parser.add_argument('-e', '--epoch', type=int, default=1, help="Number of epochs")
    return parser.parse_args()

def classification_error(): 
    #write accuracy function (y/predictin)
    pass



def main(): 
    args = parse_arguments()
    #1. Load the data (Iris Set)
    #train_set, test_set = dl.load_diab()
    #train_set, test_set = dl.load_susy()
    #train_set, test_set = dl.load_iris()
    train_set, test_set = dl.load_wine() 
    Xtrain = train_set[0]
    ytrain = train_set[1]
    Xtest = test_set[0]
    ytest = test_set[1]
   # dl.load_wine()
    #Learn the weights
    #weights = pa.fit(Xtrain, ytrain, float(sys.argv[1]))
    #yhat = pa.predict(Xtest, weights)
    #weights = sgd.fit(Xtrain, ytrain, int(sys.argv[1]))
    #yhat = sgd.predict(Xtest, weights)
    #algs = {'batch': pa.BatchGradientDescent(), 'stochastic': pa.StochasticGradientDescent()}
    #algs = {'logist': pa.LogisticRegression()}
    algs = {'softmax': pa.SoftMaxRegression()}
    params = {'tolerance': args.tolerance, 'epoch': args.epoch} #'steps': args.steps}
    #use .format for printing
    for learner_name, learner in algs.iteritems():
        print(learner_name, learner)
        learner.reset(params)
        learner.fit(Xtrain, ytrain)
        yhat = learner.predict(Xtest)
        print('yhat: {}'.format(yhat))
        print('y: {}'.format(ytest))
        print(accuracy_score(ytest, yhat))
        #print(learner.err)
        #plt.plot(learner.err)
        #plt.show()
        from sklearn.linear_model import LogisticRegression
        lg = LogisticRegression(multi_class='multinomial', solver='newton-cg', random_state=42, verbose=1, max_iter=1000, penalty="l2")
        lg.fit(Xtrain, ytrain)
        yhat = lg.predict(Xtest)
        print("----" * 50)
        print("Testing scikit logistic regression")
        print('yhat: {}'.format(yhat))
        print('y: {}'.format(ytest))
        print(accuracy_score(ytest, yhat))
        #error = np.linalg.norm(np.subtract(yhat, ytest))/ytest.shape[0]
        #print("Learner name: ", learner_name, "\n", "Error:", error)



	#batch = pa.BatchGradientDescent()
	#batch.reset(params)
	#batch.fit(Xtrain, ytrain)
	#yhat = batch.predict(Xtest)
	#print(yhat)
	#l2 norm                                                                                                                                            
	#error = np.linalg.norm(np.subtract(yhat, ytest))/ytest.shape[0]
	#print(error)
	#print("---" * 30)
	#sgd = pa.StochasticGradientDescent()
	#sgd.reset(params)
	#sgd.fit(Xtrain, ytrain)
	#yhat = sgd.predict(Xtest)
	#print(yhat)
	#l2 norm
	#error = np.linalg.norm(np.subtract(yhat, ytest))/ytest.shape[0]
	#print(error)

		
if __name__ == '__main__': 
    main()
