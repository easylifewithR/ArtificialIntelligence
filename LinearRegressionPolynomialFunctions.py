import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
data = np.loadtxt("crash.txt")

# Creating training data and testing data
trainIndex = range(0, data.shape[0], 2) # even
testIndex = range(1, data.shape[0], 2) # odd
trainData = data[trainIndex, ::]
testData = data[testIndex, ::]

poly_n = 20
rms_train = np.zeros((poly_n), dtype=float)
rms_test = np.zeros((poly_n), dtype=float)
weights = np.zeros((1, poly_n +1))
#phi_train = np.ones((trainData.shape[0], 1))
#phi_test = np.ones((testData.shape[0], 1))
for j in (np.arange(poly_n)+1):
    i = 1
    phi_train = np.ones((trainData.shape[0], 1))
    phi_test = np.ones((testData.shape[0], 1))
    while i <= j:
        phi_train = np.hstack((phi_train, np.reshape((trainData[::,0])**i, (-1, 1))))
        phi_test = np.hstack((phi_test, np.reshape((testData[::,0])**i, (-1, 1))))
        i +=1

    weights = np.linalg.solve(np.dot(np.transpose(phi_train), phi_train), np.dot(np.transpose(phi_train), trainData[::,
                                                                                                    1]))
    #weights = np.linalg.lstsq(phi_train, trainData[::, 1])[0]
    predictedVal_train = np.dot(phi_train, weights)
    predictedVal_test = np.dot(phi_test, weights)
    rms_train[j-1] = np.sqrt(np.sum((trainData[::, 1] - predictedVal_train)**2))
    rms_test[j-1] = np.sqrt(np.sum((testData[::, 1] - predictedVal_test)**2))
#print rms_train, rms_test


plt.plot(np.arange(poly_n) + 1, rms_test, label = "Test data")
plt.plot(np.arange(poly_n) + 1, rms_train, label = "Train data")
plt.xlabel("Polynomial order")
plt.ylabel("MSE")
plt.legend()
plt.show()

bestPolyOrder = np.argmin(rms_test) + 1
bestPolyOrderTrain = np.argmin(rms_train) + 1
print "Best polynomial order for testing data: ", bestPolyOrder
print "Best polynomial order for training data: ", np.argmin(rms_train) + 1

x_test = np.linspace(3, 50, 500)
phi_train = np.ones((trainData.shape[0], 1))
phi_test = np.ones((x_test.shape[0], 1))
i = 1
while i <= bestPolyOrderTrain:
    phi_train = np.hstack((phi_train, np.reshape((trainData[::, 0]) ** i, (-1, 1))))
    phi_test = np.hstack((phi_test , np.reshape((x_test) ** i, (-1, 1))))
    i += 1
weights = np.linalg.solve(np.dot(np.transpose(phi_train), phi_train), np.dot(np.transpose(phi_train), trainData[::,
                                                                                                      1]))

predictedVal = np.dot(phi_test, weights)

plt.scatter(trainData[::, 0], trainData[::, 1], label = "Train data")
plt.plot(x_test, predictedVal, label = "Best polynomial function (lowest RMS on train)", c = "r")
#plt.plot(np.linspace(0, 60, 600), np.dot(phi_train, weights), label = "Train model", c = "r")
plt.legend()
plt.show()


x_test = np.linspace(3, 50, 500)
phi_train = np.ones((trainData.shape[0], 1))
phi_test = np.ones((x_test.shape[0], 1))
i = 1
while i <= bestPolyOrder:
    phi_train = np.hstack((phi_train, np.reshape((trainData[::, 0]) ** i, (-1, 1))))
    phi_test = np.hstack((phi_test , np.reshape((x_test) ** i, (-1, 1))))
    i += 1
weights = np.linalg.solve(np.dot(np.transpose(phi_train), phi_train), np.dot(np.transpose(phi_train), trainData[::,
                                                                                                      1]))
predictedVal = np.dot(phi_test, weights)
plt.scatter(trainData[::, 0], trainData[::, 1], label = "Train data")
plt.plot(x_test, predictedVal, label = "Best polynomial function (lowest RMS on test)", c = "r")
plt.legend()
plt.show()
#print np.std(trainData[::, 0])
