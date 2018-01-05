import numpy as np
import random
from mnist import MNIST
import matplotlib.pyplot as plt
#importing MNIST data
mndata = MNIST('data')


testing_images, testing_labels = mndata.load_testing()
def kmean(noClusters, data, centroidInitial):
    data = np.asarray(data)
    #testing_labels = np.asarray(testing_labels)
    k = 0
    # Initialize input data
    #noClusters = 5
    centroidInitial = np.asarray(random.sample(data, k = noClusters)) # random.sample retured list




    # label the data based on defined mean
    labels = np.zeros((data.shape[0], noClusters))
    distFromC = np.zeros((data.shape[0], noClusters), float)
    for i in range(noClusters):
        distFromC[::, i] = np.sqrt(np.sum((data - centroidInitial[i, ::])**2, axis=1))

    temp = np.reshape(np.repeat(np.min(distFromC, axis=1), noClusters), (data.shape[0], noClusters))
    labels[(distFromC-temp == 0)] = 1


    # Finding  initial cost
    costOld = 0
    for i in range(noClusters):
        costOld = costOld + np.sum(labels[::, i]*np.sum((data - centroidInitial[i, ::])**2, axis=1))
    print costOld


    while(True):
        # Update mean
        centroid = np.zeros((noClusters, data.shape[1]))
        for i in range(noClusters):
            centroid[i, ::] = np.mean(data[labels[::, i] == 1, ::], axis=0)


        # Update label (by finding nearest centeroid)
        labels = np.zeros((data.shape[0], noClusters))
        distFromC = np.zeros((data.shape[0], noClusters), float)
        for i in range(noClusters):
            distFromC[::, i] = np.sqrt(np.sum((data - centroid[i, ::])**2, axis=1))

        nearestCentroid = np.reshape(np.repeat(np.min(distFromC, axis=1), noClusters), (data.shape[0], noClusters))
        labels[(distFromC-nearestCentroid  == 0)] = 1

        # Calculate new cost
        distFromC = np.zeros((data.shape[0], noClusters), float)
        costNew = 0
        for i in range(noClusters):
            costNew = costNew + np.sum(labels[::, i]*np.sum((data - centroid[i, ::])**2, axis=1))
        k +=1
        print costNew, k
        if costOld - costNew <= 0.01:
            break
        else:
            costOld = costNew
    return labels, centroid, costNew
# Check the change in objective function


# Selecting K number of centroid from data randomly
noClusters = 10
random.seed(1234)
centroidInitial = np.asarray(random.sample(testing_images, k = noClusters))
assignment, centroid, cost = kmean(noClusters=noClusters, data=testing_images, centroidInitial= centroidInitial)
#print cost

# Using Kmean ++ initialization
def nextIndex(data, centroid):
  #  data = np.asarray(data)
    distFromC = np.zeros((data.shape[0], centroid.shape[0]), float)
    for i in np.arange(centroid.shape[0]):
        distFromC[::, i] = np.sqrt(np.sum((data - centroid[i, ::])**2, axis=1))

    minDis = np.min(distFromC, axis=1)
    probCumSum = (minDis/minDis.sum()).cumsum()
    cut = random.random()
    return np.where(probCumSum >= cut)[0][0]

testData = np.asarray(testing_images)
centroid = np.zeros((noClusters, testData.shape[1]), float)
#centroid[0, ::] = np.asarray(random.sample(testData, k=2))
centroid[0, ::] = np.asarray(random.sample(testData, k=1))
#centroid[1, ::] = np.asarray(random.sample(testData, k=1))

for i in (np.arange(noClusters-1)):
    index = nextIndex(data=testData, centroid=centroid)
    centroid[i+1, ::] = testData[index, ::]

assignment, centroid, cost = kmean(noClusters=noClusters, data=testing_images, centroidInitial= centroid)
print "End of Part 1"

#centroid = np.asarray(random.sample(testData, k = noClusters))
# plt.subplot(520)
# pixel = centroid[0, ::].reshape((28,28))
# #plt.title("Correct prediciton")
# plt.imshow(pixel, cmap=plt.get_cmap('gray'))
#
# plt.subplot(521)
# pixel = centroid[1, ::].reshape((28,28))
# #plt.title("Correct prediciton")
# plt.imshow(pixel, cmap=plt.get_cmap('gray'))
# plt.subplot(522)
# pixel = centroid[2, ::].reshape((28,28))
# #plt.title("Correct prediciton")
# plt.imshow(pixel, cmap=plt.get_cmap('gray'))
# plt.subplot(523)
# pixel = centroid[3, ::].reshape((28,28))
# #plt.title("Correct prediciton")
# plt.imshow(pixel, cmap=plt.get_cmap('gray'))
# plt.subplot(524)
# pixel = centroid[4, ::].reshape((28,28))
# #plt.title("Correct prediciton")
# plt.imshow(pixel, cmap=plt.get_cmap('gray'))
# plt.subplot(525)
# pixel = centroid[5, ::].reshape((28,28))
# #plt.title("Correct prediciton")
# plt.imshow(pixel, cmap=plt.get_cmap('gray'))
# plt.subplot(526)
# pixel = centroid[6, ::].reshape((28,28))
# #plt.title("Correct prediciton")
# plt.imshow(pixel, cmap=plt.get_cmap('gray'))
# plt.subplot(527)
# pixel = centroid[7, ::].reshape((28,28))
# #plt.title("Correct prediciton")
# plt.imshow(pixel, cmap=plt.get_cmap('gray'))
# plt.subplot(528)
# pixel = centroid[8, ::].reshape((28,28))
# #plt.title("Correct prediciton")
# plt.imshow(pixel, cmap=plt.get_cmap('gray'))
# plt.subplot(529)
# pixel = centroid[9, ::].reshape((28,28))
# #plt.title("Correct prediciton")
# plt.imshow(pixel, cmap=plt.get_cmap('gray'))
#
#
# plt.show()


# Centers from test data
#indexCheating = np.zeros((9))
testLabels = np.asarray(testing_labels)
centroid = np.zeros((noClusters, testData.shape[1]))
for i in np.arange(9):
    centroid[i, ::] = testData[np.where(testLabels == i)[0][0],::]

assignment, centroid, cost = kmean(noClusters=noClusters, data=testing_images, centroidInitial= centroid)

print "End of Part 2"

# plt.subplot(520)
# pixel = centroid[0, ::].reshape((28,28))
# plt.imshow(pixel, cmap=plt.get_cmap('gray'))
#
# plt.subplot(521)
# pixel = centroid[1, ::].reshape((28,28))
# plt.imshow(pixel, cmap=plt.get_cmap('gray'))
# plt.subplot(522)
# pixel = centroid[2, ::].reshape((28,28))
# plt.imshow(pixel, cmap=plt.get_cmap('gray'))
# plt.subplot(523)
# pixel = centroid[3, ::].reshape((28,28))
# plt.imshow(pixel, cmap=plt.get_cmap('gray'))
# plt.subplot(524)
# pixel = centroid[4, ::].reshape((28,28))
# plt.imshow(pixel, cmap=plt.get_cmap('gray'))
# plt.subplot(525)
# pixel = centroid[5, ::].reshape((28,28))
#
# plt.imshow(pixel, cmap=plt.get_cmap('gray'))
# plt.subplot(526)
# pixel = centroid[6, ::].reshape((28,28))
#
# plt.imshow(pixel, cmap=plt.get_cmap('gray'))
# plt.subplot(527)
# pixel = centroid[7, ::].reshape((28,28))
#
# plt.imshow(pixel, cmap=plt.get_cmap('gray'))
# plt.subplot(528)
# pixel = centroid[8, ::].reshape((28,28))
#
# plt.imshow(pixel, cmap=plt.get_cmap('gray'))
# plt.subplot(529)
# pixel = centroid[9, ::].reshape((28,28))
# plt.imshow(pixel, cmap=plt.get_cmap('gray'))
# plt.show()


#k = 3 using K-Mean
noClusters2 = 3
testData = np.asarray(testing_images)
centroid = np.zeros((noClusters2, testData.shape[1]), float)
#centroid[0, ::] = np.asarray(random.sample(testData, k=2))
centroid[0, ::] = np.asarray(random.sample(testData, k=1))
#centroid[1, ::] = np.asarray(random.sample(testData, k=1))

for i in (np.arange(noClusters2-1)):
    index = nextIndex(data=testData, centroid=centroid)
    centroid[i+1, ::] = testData[index, ::]

assignment, centroid, cost = kmean(noClusters=noClusters, data=testing_images, centroidInitial= centroid)
print "End of Part 3"

plt.subplot(310)
pixel = centroid[0, ::].reshape((28,28))
plt.imshow(pixel, cmap=plt.get_cmap('gray'))

plt.subplot(311)
pixel = centroid[1, ::].reshape((28,28))
plt.imshow(pixel, cmap=plt.get_cmap('gray'))
plt.subplot(312)
pixel = centroid[2, ::].reshape((28,28))
plt.imshow(pixel, cmap=plt.get_cmap('gray'))
plt.show()

temp0 = testData[np.where(assignment[::, 0])[0][0], ::]
plt.subplot(521)
pixel = temp0.reshape((28,28))
plt.imshow(pixel, cmap=plt.get_cmap('gray'))


temp1 = testData[np.where(assignment[::, 1])[0][0], ::]
plt.subplot(522)
pixel = temp1.reshape((28,28))
plt.imshow(pixel, cmap=plt.get_cmap('gray'))

temp2 = testData[np.where(assignment[::, 2])[0][0], ::]
plt.subplot(523)
pixel = temp2.reshape((28,28))
plt.imshow(pixel, cmap=plt.get_cmap('gray'))

temp3 = testData[np.where(assignment[::, 3])[0][0], ::]
plt.subplot(524)
pixel = temp3.reshape((28,28))
plt.imshow(pixel, cmap=plt.get_cmap('gray'))


temp4 = testData[np.where(assignment[::, 4])[0][0], ::]
plt.subplot(525)
pixel = temp4.reshape((28,28))
plt.imshow(pixel, cmap=plt.get_cmap('gray'))


temp5 = testData[np.where(assignment[::, 5])[0][0], ::]
plt.subplot(526)
pixel = temp5.reshape((28,28))
plt.imshow(pixel, cmap=plt.get_cmap('gray'))

temp6 = testData[np.where(assignment[::, 6])[0][0], ::]
plt.subplot(527)
pixel = temp6.reshape((28,28))
plt.imshow(pixel, cmap=plt.get_cmap('gray'))

plt.show()
print "Yahoo! Got it"
