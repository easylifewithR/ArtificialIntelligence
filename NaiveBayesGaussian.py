from __future__ import division #for division (/)
import numpy as np
from scipy.stats import norm
from sklearn.metrics import confusion_matrix #(for confusion matrix)
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB # Just for testing the implementation (remove it)


from numpy import random
from mnist import MNIST

mndata = MNIST('data')
training_images, training_labels = mndata.load_training()
testing_images, testing_labels = mndata.load_training()
#print images
training_images = np.asarray(training_images)
training_labels = np.asarray(training_labels)

m = 1000 #sample size
data_5 = training_images[training_labels == 5, ::] # sampling from training data for which label is 5
data_not5 = training_images[np.logical_not(training_labels == 5), ::]
data_5 = np.hstack((data_5, np.ones((data_5.shape[0], 1))))
data_not5 = np.hstack(((data_not5, np.zeros((data_not5.shape[0], 1)))))

np.random.seed(12345)
index = np.random.choice(np.arange(data_5.shape[0]), m, replace= False)
# dividing the data into training and testing data
trainData_5 = data_5[index[0:m*0.9], :]
testData_5 = data_5[index[m*0.9:m], :]

np.random.seed(12345)
index = np.random.choice(np.arange(data_not5.shape[0]), m, replace = False)
trainData_not5 = data_not5[index[0:m*0.9], :]
testData_not5 = data_not5[index[m*0.9:m], :]

trainData = np.vstack((trainData_5, trainData_not5))
testData = np.vstack((testData_5, testData_not5))

np.random.shuffle(trainData)
np.random.shuffle(testData)
# print trainData.shape
# print testData.shape
# np.savetxt("trainData.csv", trainData, delimiter=",")
# np.savetxt("testData.csv", testData, delimiter=",")

#trainData = np.genfromtxt("trainData.csv", delimiter=",")
#testData = np.genfromtxt("testData.csv", delimiter=",")

trainFeautures_y0 = trainData[trainData[::, (trainData.shape[1]-1)] == 0, :(trainData.shape[1]-1):1]
trainFeautures_y1 = trainData[trainData[::, (trainData.shape[1]-1)] == 1, :(trainData.shape[1]-1):1]
meanFeatures_y0 = np.mean(trainFeautures_y0, axis=0)
meanFeatures_y1 = np.mean(trainFeautures_y1 , axis=0)

var_y0 = np.true_divide(np.sum((trainFeautures_y0 - meanFeatures_y0)**2), trainFeautures_y0.shape[
    0]*trainFeautures_y0.shape[1])

var_y1 = np.true_divide(np.sum((trainFeautures_y1 - meanFeatures_y1)**2), trainFeautures_y1.shape[
    0]*trainFeautures_y1.shape[1])

testFeatures = testData[::, :(testData.shape[1]-1):1]
likelihood_y0 = np.sum(np.log(norm.pdf(np.true_divide(testFeatures-meanFeatures_y0, np.sqrt(var_y0)))),
                       axis=1) + np.log(0.5*np.ones(testData.shape[0]))

likelihood_y1 = np.sum(np.log(norm.pdf(np.true_divide(testFeatures-meanFeatures_y1, np.sqrt(var_y1)))),
                       axis = 1) + + np.log(0.5*np.ones(testData.shape[0]))

#featureMeans = np.mean(trainData[::, :(trainData.shape[1]-1):1], axis=0)

#featureVars = np.true_divide(np.sum((trainData[::, :(trainData.shape[1]-1):1] - featureMeans)**2), trainData[::,
# :(trainData.shape[1]-1):1].shape[
#    0]*trainData[::, :(trainData.shape[1]-1):1].shape[1])
#norm_testFeatures = np.sum(np.log(norm.pdf(np.true_divide((testFeatures-featureMeans), np.sqrt(featureVars)))), axis=1)

#posterior_0 = likelihood_y0 - norm_testFeatures
#posterior_1 = likelihood_y1 - norm_testFeatures
# likelihood_y0 = np.nan_to_num(likelihood_y0)
#likelihood_y0 = np.log(np.true_divide(np.exp(likelihood_y0), np.sqrt(2*np.pi*(sdFeatures_y0)**2)))
#likelihood_y0 = np.log(likelihood_y0)
#likelihood_y0 = np.sum(likelihood_y0, axis=1) + np.log(0.5*np.ones(testData.shape[0]))

# # posterior = np.multiply(likelihood_y0, np.repeat(0.5, likelihood_y0.shape[0]))

predictedValue = np.zeros((testData.shape[0]))
predictedValue[likelihood_y0 <= likelihood_y1]=1
acc = (testData[::, (testData.shape[1]-1)]== predictedValue).mean()
print "Clasification accuracy (gaussian naive bayes)", acc


loglikelihoodDiff = likelihood_y1-likelihood_y0




plt.figure()
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")


for costRatio in np.array([5, 2, 1, 0.5, 0.2]):
    prediction  = np.zeros((testData.shape[0]))
    prediction[loglikelihoodDiff  >= np.log(costRatio)] =  1
    cm = confusion_matrix(y_true= testData[::, (testData.shape[1] -1)], y_pred= prediction)
    tpr = cm[1, 1]/(cm[1, 0] + cm[1, 1])
    fpr = cm[0,1]/(cm[0,0] + cm[0,1])
    print "True positive rate (CostType1/CostType2 = ",  costRatio, "): ", tpr
    print "False positive rate (CostType1/CostType2 = ",  costRatio, "): ", fpr




plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")

tpr = np.zeros((10))
fpr = np.zeros((10))
i = 1
for costRatio in np.array([5, 2, 1, 0.5, 0.2, 1e-20, 1e-30, 1e-50]):
    prediction  = np.zeros((testData.shape[0]))
    prediction[loglikelihoodDiff  >= np.log(costRatio)] =  1
    cm = confusion_matrix(y_true= testData[::, (testData.shape[1] -1)], y_pred= prediction)
    tpr[i] = cm[1, 1]/(cm[1, 0] + cm[1, 1])
    fpr[i] = cm[0,1]/(cm[0,0] + cm[0,1])
    #print "True positive rate (CostType1/CostType2 = ",  costRatio, "): ", tpr[i]
    #print "False positive rate (CostType1/CostType2 = ",  costRatio, "): ", fpr[i]
    i +=1
fpr[i] = 1
tpr[i] = 1
plt.plot(fpr, tpr, color = "red")
plt.title("Receiver operating characteristic curve")
plt.show()