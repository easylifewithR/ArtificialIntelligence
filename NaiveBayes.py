import numpy as np
from mnist import MNIST
#from sklearn.naive_bayes import MultinomialNB

mndata = MNIST('data')
training_images, training_labels = mndata.load_training()
testing_images, testing_labels = mndata.load_training()
#print images
training_images = np.asarray(training_images)
training_labels = np.asarray(training_labels)
training_images[training_images < 200] = 0
training_images[np.logical_not(training_images < 200)] = 1

testing_images = np.asarray(testing_images)
testing_labels = np.asarray(testing_labels)
testing_images[testing_images < 200] = 0
testing_images[np.logical_not(testing_images < 200)] = 1

n_class = len(np.unique(training_labels)) #number of levels in class variable
n_features  = training_images.shape[1] #number of features
# Used to cross validate the answer
# mnb = MultinomialNB()
# y_pred_mnb = mnb.fit(training_images, training_labels).predict(testing_images)
# print (y_pred_mnb == testing_labels).mean()
# Calculating the likelihood (parameters)
theta = np.zeros((n_class, n_features))
for k in range(n_class):
    theta[k,::] = np.true_divide((np.sum(training_images[training_labels == k, ::], axis=0) + 1),
                                 (np.sum(training_labels == k) + n_class))
# Prior

prior = np.repeat(np.true_divide(1, n_class), n_class) # 10 is number of outcome levels
predictedValue = np.zeros((testing_images.shape[0]))
for i in range(testing_images.shape[0]):
    temp = testing_images[i, ::]
    likelihood = np.zeros((n_class, n_features))
    likelihood[::, temp == 0] = 1 - theta[::, temp == 0]
    likelihood[::, temp == 1] = theta[::, temp == 1]
    loglikelihood = np.sum(np.log(likelihood), axis=1) + np.log(prior)
    predictedValue[i] = np.argmax(loglikelihood)
print "Clasification accuracy (naive bayes)", (predictedValue == testing_labels).mean()



