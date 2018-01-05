import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt("crash.txt")

# Creating training data and testing data
trainIndex = range(0, data.shape[0], 2) # even
testIndex = range(1, data.shape[0], 2) # odd
trainData = data[trainIndex, ::]
testData = data[testIndex, ::]

noBasisFunctions = np.array([5, 10, 15, 20, 25])
xMin = np.min(trainData[::, 0])
xMax = np.max(trainData[::, 0])
rms_train = np.zeros((noBasisFunctions.size), dtype=float)
rms_test = np.zeros((noBasisFunctions.size), dtype=float)
j = 0
for i in noBasisFunctions:
    stepSize = (xMax - xMin)/(i +1)
    # repeating the colum to create radial features
    train_xTransformed = np.repeat(trainData[::, 0], i)
    train_xTransformed = train_xTransformed.reshape(trainData.shape[0], i)
    test_xTransformed = np.repeat(testData[::, 0], i)
    test_xTransformed = test_xTransformed.reshape(testData.shape[0], i)

# picking the mean for radial basis function
    xMean = np.arange(xMin,xMax,stepSize)
    xMean = xMean[0:i:]
    xMean = xMean + stepSize/2.0
    #xMean = xMean[1::]
    sd = np.array(xMean[1] - xMean[0])
# Creating the featues by transforming to
    train_xTransformed = np.exp(-1*((train_xTransformed - xMean)**2/(2*sd**2)))
    train_xTransformed = np.hstack((np.ones((trainData.shape[0], 1), float), train_xTransformed))
    test_xTransformed = np.exp(-1*((test_xTransformed - xMean)**2/(2*sd**2)))
    test_xTransformed = np.hstack((np.ones((testData.shape[0], 1), float), test_xTransformed))
    weights = np.linalg.solve(np.dot(np.transpose(train_xTransformed),train_xTransformed), np.dot(np.transpose(train_xTransformed), trainData[::,1]))
    predictedVal_train = np.dot(train_xTransformed, weights)
    predictedVal_test = np.dot(test_xTransformed, weights)
    rms_train[j] = np.sqrt(np.sum((trainData[::, 1] - predictedVal_train)** 2))
    rms_test[j] = np.sqrt(np.sum((testData[::, 1] - predictedVal_test)** 2))
    j +=1

#print rms_train, rms_test

plt.plot(noBasisFunctions, rms_train, label = "Train data")
plt.plot(noBasisFunctions, rms_test, label = "Test data")

#plt.plot(np.linspace(xMin, xMax), rms_train, label = "Train data")
plt.xlabel("Number of radial functions")
plt.ylabel("MSE")
plt.legend()
plt.show()

bestRadialBasis = noBasisFunctions[np.argmin(rms_test)]
bestRadialBasisTrain = noBasisFunctions[np.argmin(rms_train)]
print "Optimal number of radial functions (Testing data): ", bestRadialBasis
print "Optimal number of radial functions (Training data): ", bestRadialBasisTrain


stepSize = (xMax - xMin)/(bestRadialBasisTrain +1)
    # repeating the colum to create radial features
train_xTransformed = np.repeat(trainData[::, 0], bestRadialBasisTrain)
train_xTransformed = train_xTransformed.reshape(trainData.shape[0], bestRadialBasisTrain)
testGen = np.linspace(3, 50, 500)
test_xTransformed = np.repeat(testGen, bestRadialBasisTrain)
test_xTransformed = test_xTransformed.reshape(testGen.shape[0], bestRadialBasisTrain)

# picking the mean for radial basis function
xMean = np.arange(xMin,xMax,stepSize)
xMean = xMean[0:bestRadialBasisTrain:]
xMean = xMean + stepSize/2.0
#xMean = xMean[1::]
sd = np.array(xMean[1] - xMean[0])
# Creating the featues by transforming to
train_xTransformed = np.exp(-1*((train_xTransformed - xMean)**2/(2*sd**2)))
train_xTransformed = np.hstack((np.ones((trainData.shape[0], 1), float), train_xTransformed))
test_xTransformed = np.exp(-1*((test_xTransformed - xMean)**2/(2*sd**2)))
test_xTransformed = np.hstack((np.ones((testGen.shape[0], 1), float), test_xTransformed))
weights = np.linalg.solve(np.dot(np.transpose(train_xTransformed),train_xTransformed), np.dot(np.transpose(train_xTransformed), trainData[::,1]))
#predictedVal_train = np.dot(train_xTransformed, weights)
predictedVal = np.dot(test_xTransformed, weights)
#rms_train = np.sqrt(np.sum((trainData[::, 1] - predictedVal_train)** 2))
#rms_test = np.sqrt(np.sum((testData[::, 1] - predictedVal_test)** 2))


plt.scatter(trainData[::, 0], trainData[::, 1], label = "Train data")
plt.plot(testGen, predictedVal, label = "Radial function model (lowest RMS on train)", c = "r")
plt.legend()
plt.show()


stepSize = (xMax - xMin)/(bestRadialBasis +1)
    # repeating the colum to create radial features
train_xTransformed = np.repeat(trainData[::, 0], bestRadialBasis)
train_xTransformed = train_xTransformed.reshape(trainData.shape[0], bestRadialBasis)
testGen = np.linspace(3, 50, 500)
test_xTransformed = np.repeat(testGen, bestRadialBasis)
test_xTransformed = test_xTransformed.reshape(testGen.shape[0], bestRadialBasis)

# picking the mean for radial basis function
xMean = np.arange(xMin,xMax,stepSize)
xMean = xMean[0:bestRadialBasis:]
xMean = xMean + stepSize/2.0
#xMean = xMean[1::]
sd = np.array(xMean[1] - xMean[0])
# Creating the featues by transforming to
train_xTransformed = np.exp(-1*((train_xTransformed - xMean)**2/(2*sd**2)))
train_xTransformed = np.hstack((np.ones((trainData.shape[0], 1), float), train_xTransformed))
test_xTransformed = np.exp(-1*((test_xTransformed - xMean)**2/(2*sd**2)))
test_xTransformed = np.hstack((np.ones((testGen.shape[0], 1), float), test_xTransformed))
weights = np.linalg.solve(np.dot(np.transpose(train_xTransformed),train_xTransformed), np.dot(np.transpose(train_xTransformed), trainData[::,1]))
#predictedVal_train = np.dot(train_xTransformed, weights)
predictedVal = np.dot(test_xTransformed, weights)
#rms_train = np.sqrt(np.sum((trainData[::, 1] - predictedVal_train)** 2))
#rms_test = np.sqrt(np.sum((testData[::, 1] - predictedVal_test)** 2))


plt.scatter(trainData[::, 0], trainData[::, 1], label = "Train data")
plt.plot(testGen, predictedVal, label = "Radial function model (lowest RMS on test)", c = "r")
plt.legend()
plt.show()




