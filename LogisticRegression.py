import numpy as np
from scipy.optimize import minimize
import random

def flower_to_float(s):
    d = { "Iris-setosa" :0., "Iris-versicolor" :1., "Iris-virginica" :2.}
    return d[s]

irises = np.loadtxt("iris.data.txt", delimiter= "," , converters={4:flower_to_float})
n_features = irises.shape[1] -1 # excluding target variable

n_levels = 3 # Number of classes in outcome variables
irises = np.hstack((irises, np.reshape(irises[::, (irises.shape[1]-1)], (-1, 1)), np.reshape(irises[::, (irises.shape[
                                                                                                           1]-1)],
                  (-1, 1))))
irises[np.logical_not(irises[::,(irises.shape[1]-1)] == 2), irises.shape[1]-1] = 0
irises[irises[::,(irises.shape[1]-1)] == 2, irises.shape[1]-1] = 1
irises[np.logical_not(irises[::,(irises.shape[1]-2)] == 1), irises.shape[1]-2] = 0
irises[np.logical_not(irises[::,(irises.shape[1]-3)] == 0), irises.shape[1]-3] = 1
irises[::,(irises.shape[1]-3)] = np.logical_not(irises[::,(irises.shape[1]-3)])

irises = np.hstack((np.reshape(np.ones((irises.shape[0])), (-1, 1)), irises))

trainIndex = np.arange(0,irises.shape[0], 2)
testIndex = np.arange(1,irises.shape[0], 2)
irisesTrain = irises[trainIndex, ::]
irisesTest = irises[testIndex, ::]

j = 0
alpha = 0.03
weights = np.ones((n_features + 1)*n_levels) # with bias
#weights = np.repeat(0.001, 15)
def f(weights):
    obj_part1 = alpha/2.0*np.dot(weights, weights)
    obj_part2 = np.sum(irisesTrain[::, n_features+1::]*np.dot(irisesTrain[::, :(n_features+1):], np.transpose(
        np.reshape(weights,
                                                                                                      (3, 5)))))
    obj_part3 = np.sum(np.log(np.sum(np.exp(np.dot(irisesTrain[::, :(n_features+1):], np.transpose(np.reshape(weights,
                                                                                                             (3,
                                                                                                            5))))), axis = 1)), axis=0)
    obj = obj_part1 - obj_part2 + obj_part3
    #print obj
    return obj


w_hat = minimize(f, weights).x
print w_hat
predictedRegVal = np.dot(irisesTest[::, :(n_features + 1):] , np.transpose(np.reshape(w_hat, (3, 5))))

den_posteriors = np.sum(np.exp(predictedRegVal), axis=1)
posterior = np.exp(predictedRegVal)/np.reshape(den_posteriors, (-1,1))
predictedVal = np.argmax(posterior, axis = 1)
acutalOutcome = np.argmax(irisesTest[::, (n_features + 1)::], axis=1)
acc = (predictedVal == acutalOutcome).mean()
#/den_posteriors

print "Classification accuracy:", acc