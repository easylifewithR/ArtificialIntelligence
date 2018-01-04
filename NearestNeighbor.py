from scipy.stats import mode
import numpy as np
from mnist import MNIST
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

mndata = MNIST('data')
training_images, training_labels = mndata.load_training()
testing_images, testing_labels = mndata.load_training()
#print images
training_images = np.asarray(training_images)
training_labels = np.asarray(training_labels)

testing_images = np.asarray(testing_images)
testing_labels = np.asarray(testing_labels)
#Following is for Part 3


#Create dataset 1, 2, 7
m_train = 200 # sample train data size for each class
m_test = 50
#data_1 = training_images[training_labels == 1, :]
#data_2 = training_images[training_labels == 2, :]
#data_7 = training_images[training_labels == 7, :]

np.random.seed(12345)
index = np.random.choice(np.arange(training_images[training_labels == 1, :].shape[0]), m_train, replace= False)
trainData_1 = training_images[training_labels == 1, :][index, ::]
index = np.random.choice(np.arange(testing_images[testing_labels == 1, :].shape[0]), m_test, replace= False)
testData_1 = testing_images[testing_labels == 1, :][index, ::]

#data_1 = np.hstack((selData_1, np.ones((selData_1.shape[0], 1))))
index = np.random.choice(np.arange(training_images[training_labels == 2, :].shape[0]), m_train, replace= False)
trainData_2 = training_images[training_labels == 2, :][index, ::]

index = np.random.choice(np.arange(testing_images[testing_labels == 2, :].shape[0]), m_test, replace= False)
testData_2 = testing_images[testing_labels == 2, :][index, ::]
#data_2 = np.hstack((selData_2, 2*np.ones((selData_2.shape[0], 1))))

index = np.random.choice(np.arange(training_images[training_labels == 7, :].shape[0]), m_train, replace= False)
trainData_7 = training_images[training_labels == 7, :][index, ::]

index = np.random.choice(np.arange(testing_images[testing_labels == 7, :].shape[0]), m_test, replace= False)
testData_7 = testing_images[testing_labels == 7, :][index, ::]

#data_7 = np.hstack((selData_7, 7*np.ones((selData_7.shape[0], 1))))
trainData = np.vstack((np.hstack((trainData_1, np.ones((trainData_1.shape[0], 1)))), np.hstack((trainData_2, 2*np.ones((
    trainData_2.shape[0], 1)))), np.hstack((trainData_7, 7*np.ones((trainData_7.shape[0], 1))))))


testData = np.vstack((np.hstack((testData_1, np.ones((testData_1.shape[0], 1)))), np.hstack((testData_2, 2*np.ones((
    testData_2.shape[0], 1)))), np.hstack((testData_7, 7*np.ones((testData_7.shape[0], 1))))))

np.random.shuffle(trainData) # to randomize the train Data
np.random.shuffle(testData)


#==Finding the optimal number of neibhors
candidate_neibhors = np.array([1, 3, 5, 7, 9])
accuCandidateNeibhor = np.zeros((candidate_neibhors.shape[0]))
j = 0
for no_neibhors in candidate_neibhors:
   # no_neibhors = 1
    k = 5# 5K cross validation
    kf = KFold(n_splits=k)
    fold = 0
    foldAcc = np.zeros((k))
    for train_idx, test_idx in kf.split(trainData):
        fold += 1
        X_train, X_test = trainData[train_idx, 0:(trainData.shape[1]-1):], trainData[test_idx, 0:(trainData.shape[1]-1):]
        y_train, y_test = trainData[train_idx, (testData.shape[1]-1)], trainData[test_idx, (trainData.shape[1]-1)]
        prediction = np.zeros((X_test.shape[0]))
        for i in np.arange(X_test.shape[0]):
            dist  =  np.sqrt(np.sum((X_train - X_test[i, ::])**2, axis = 1))
            nearestNeibhorsLabels = y_train[np.argsort(dist)[0:no_neibhors]]
            prediction[i] = mode(nearestNeibhorsLabels)[0]
    #    print prediction == y_test
        foldAcc[fold-1] = (prediction == y_test).mean()
        #print "Fold(", fold, ")", (prediction == y_test).mean()
    accuCandidateNeibhor[j] = foldAcc.mean()
    j +=1
    print no_neibhors, "-Nearest neibhor validation data accuracy:", foldAcc.mean()
optimal_neibhors = candidate_neibhors[np.argmax(accuCandidateNeibhor)]
print "Optimal number of nearest neibhors: ", candidate_neibhors[np.argmax(accuCandidateNeibhor)]
#----End of finding the optimal number of neibhor

prediction = np.zeros((testData.shape[0]))
for i in np.arange(testData.shape[0]):
    dist = np.sqrt(np.sum((trainData[::,  0:(trainData.shape[1]-1):] - testData[i, 0:(testData.shape[1]-1):]) ** 2,
                          axis=1))
    nearestNeibhorsLabels = trainData[np.argsort(dist)[0:optimal_neibhors], (testData.shape[1]-1)]
    prediction[i] = mode(nearestNeibhorsLabels)[0]
print "Test set accuracy: ",(prediction == testData[::, (testData.shape[1]-1)]).mean()
# dividing the data into training and testing data


testData_1index = testData[::, (testData.shape[1]-1)] == 1
testData_2index = testData[::, (testData.shape[1]-1)] == 2
testData_7index = testData[::, (testData.shape[1]-1)] == 7
testing1_data = testData[testData_1index, 0:(trainData.shape[1]-1):]
testing2_data = testData[testData_2index, 0:(trainData.shape[1]-1):]
testing7_data = testData[testData_7index, 0:(trainData.shape[1]-1):]

testingClass1_correct = testing1_data[prediction[testData_1index] == 1]
testingClass2_correct = testing2_data[prediction[testData_2index] == 2]
testingClass7_correct = testing7_data[prediction[testData_7index] == 7]


testingClass1_incorrect = testing1_data[prediction[testData_1index] != 1]
testingClass2_incorrect = testing2_data[prediction[testData_2index] != 2]
testingClass7_incorrect = testing7_data[prediction[testData_7index] != 7]

plt.subplot(231)
pixel = testingClass2_correct[0].reshape((28,28))
plt.title("Correct prediciton")
plt.imshow(pixel, cmap=plt.get_cmap('gray'))

plt.subplot(232)
pixel = testingClass2_correct[1].reshape((28,28))
plt.title("Correct prediciton")
plt.imshow(pixel, cmap=plt.get_cmap('gray'))


plt.subplot(233)
pixel = testingClass2_correct[2].reshape((28,28))
plt.title("Correct prediciton")
plt.imshow(pixel, cmap=plt.get_cmap('gray'))

plt.subplot(234)
pixel = testingClass2_incorrect[0].reshape((28,28))
plt.title("Incorrect prediction")
plt.imshow(pixel, cmap=plt.get_cmap('gray'))

plt.subplot(235)
pixel = testingClass2_incorrect[1].reshape((28,28))
plt.title("Incorrect prediction")
plt.imshow(pixel, cmap=plt.get_cmap('gray'))

plt.subplot(236)
pixel = testingClass2_incorrect[2].reshape((28,28))
plt.title("Incorrect prediction")
plt.imshow(pixel, cmap=plt.get_cmap('gray'))
plt.show()

plt.subplot(231)
pixel = testingClass1_correct[0].reshape((28,28))
plt.title("Correct prediciton")
plt.imshow(pixel, cmap=plt.get_cmap('gray'))

plt.subplot(232)
pixel = testingClass1_correct[1].reshape((28,28))
plt.title("Correct prediciton")
plt.imshow(pixel, cmap=plt.get_cmap('gray'))


plt.subplot(233)
pixel = testingClass1_correct[2].reshape((28,28))
plt.title("Correct prediciton")
plt.imshow(pixel, cmap=plt.get_cmap('gray'))

plt.subplot(234)
pixel = testingClass1_incorrect[0].reshape((28,28))
plt.title("Incorrect prediction")
plt.imshow(pixel, cmap=plt.get_cmap('gray'))
plt.show()
