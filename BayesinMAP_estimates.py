import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt("crash.txt")
trainIndex = range(0, data.shape[0], 2) # even
testIndex = range(1, data.shape[0], 2) # odd
trainData = data[trainIndex, ::]
testData = data[testIndex, ::]

noBasisFunctions = 50
# Model
xMax = np.max(trainData[::, 0])
xMin = np.min(trainData[::, 0])
stepSize = (xMax - xMin)/(noBasisFunctions +1)
# repeating the colum to create radial features
train_xTransformed = np.repeat(trainData[::, 0], noBasisFunctions)
train_xTransformed = train_xTransformed.reshape(trainData.shape[0], noBasisFunctions)
test_xTransformed = np.repeat(testData[::, 0], noBasisFunctions)
test_xTransformed = test_xTransformed.reshape(testData.shape[0], noBasisFunctions)


# picking the mean for radial basis function
xMean = np.arange(xMin,xMax,stepSize)
xMean = xMean[0:noBasisFunctions:] # we have noBasisFunctions + 1 segments
xMean = xMean + stepSize/2.0
#xMean = xMean[1::]
sd = np.array(xMean[1] - xMean[0])
# Creating the featues by transforming to
train_xTransformed = np.exp(-1*(train_xTransformed - xMean)**2/(2*sd**2))
train_xTransformed = np.hstack((np.ones((trainData.shape[0], 1), float), train_xTransformed))
test_xTransformed = np.exp(-1*(test_xTransformed - xMean)**2/(2*sd**2))
test_xTransformed = np.hstack((np.ones((testData.shape[0], 1), float), test_xTransformed))


# Bayesian parameter estimation
alpha = np.logspace(-8, 0, 100) # prior distribution parameter
beta = 0.0025 # estimate of inverse of variance calculated from data
rms = np.zeros((np.size(alpha)))
for i in range(np.size(alpha)):
    a = np.dot(np.transpose(train_xTransformed), train_xTransformed) + (alpha[i]/beta)*np.identity(
        train_xTransformed.shape[1])
    b = np.dot(np.transpose(train_xTransformed), trainData[::, 1])
    weights = np.linalg.solve(a, b)
    predictedVal = np.dot(test_xTransformed, weights)
    rms[i] = np.sum((predictedVal - testData[::, 1])**2)
best_alpha = alpha[np.argmin(rms)]

print "Best alpha: ", best_alpha

a = np.dot(np.transpose(train_xTransformed), train_xTransformed) + (best_alpha/beta)*np.identity(
    train_xTransformed.shape[1])
b = np.dot(np.transpose(train_xTransformed), trainData[::, 1])
weights = np.linalg.solve(a, b)

testGen = np.linspace(3, 50, 500)
test_xTransformed = np.repeat(testGen, noBasisFunctions)
test_xTransformed = test_xTransformed.reshape(testGen.shape[0], noBasisFunctions)
test_xTransformed = np.exp(-1*(test_xTransformed - xMean)**2/(2*sd**2))
test_xTransformed = np.hstack((np.ones((testGen.shape[0], 1), float), test_xTransformed))

predictedVal = np.dot(test_xTransformed, weights)


plt.scatter(data[::, 0], data[::, 1], c = "g", label = "Observed data")
plt.plot(testGen, predictedVal, c = "r", label = "Function using Bayesian parameter estimation")
plt.legend()
plt.show()