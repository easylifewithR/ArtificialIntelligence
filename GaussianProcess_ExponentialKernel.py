import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
data = np.loadtxt("crash.txt")


def kernel_exp(x, y, kernalParameter):
    dist = np.zeros((x.shape[0], y.shape[0]))
    for i in np.arange(y.shape[0]):
        dist[::, i] = x[::,0] - y[i, ::]
    return np.exp(-1 * np.abs(dist) /kernalParameter)

varMax = np.max(data, axis=0)
normalizedData = data/varMax
kgram = pdist(normalizedData, metric='euclidean')
beta = 0.0025/varMax[1]


# Estimation of neibhorhood of kernal parameter
kernalParameter = 3000
kernel = kernel_exp(normalizedData[::, 0].reshape(-1,1),np.reshape(normalizedData[::, 0], (-1, 1)), kernalParameter)


noise = np.identity(normalizedData.shape[0])*beta
cKernel = kernel + noise
L = np.linalg.cholesky(cKernel)

# Generate evenly spaced x axis for test
synTestData = np.linspace(0, 1, 50).reshape(-1,1)

# Find mean
fac1 = np.linalg.solve(L, kernel_exp(normalizedData[::, 0].reshape(-1, 1), synTestData, kernalParameter))
fac2 = np.linalg.solve(L, normalizedData[::, 1])
mu = np.dot(np.transpose(fac1), fac2)




# Inspection
plt.scatter(normalizedData[::, 0], normalizedData[::, 1])
plt.plot(synTestData, mu, 'r')
plt.title("Exponential Kernel Parameter (to test the hyper-parameter)")


plt.show()
i = 0
kernalParameterRarnge = np.linspace(3000, 5000, 100)
meanError = np.zeros((100))
for kernalParameter in kernalParameterRarnge:
    k = 5  # 5K cross validation
    error = np.zeros((k))
    kf = KFold(n_splits=k)
    fold = 0
    foldAcc = np.zeros((k))
    for train_idx, test_idx in kf.split(normalizedData):

        X_train, X_test = normalizedData[train_idx, 0:(normalizedData.shape[1] - 1):], normalizedData[test_idx,
                                                                             0:(normalizedData.shape[1] - 1):]
        y_train, y_test = normalizedData[train_idx, (normalizedData.shape[1] - 1)], normalizedData[test_idx, (normalizedData.shape[1] - 1)]
        kernel = kernel_exp(X_train, X_train, kernalParameter)
        noise = np.identity(X_train.shape[0]) * beta
        cKernel = kernel + noise
        # compute the mean at our test points.
        L = np.linalg.cholesky(cKernel)
        fac1 = np.linalg.solve(L, kernel_exp(X_train, X_test, kernalParameter))
        fac2 = np.linalg.solve(L, y_train)
        mu = np.dot(fac1.T, fac2)
        error[fold] = np.sum((y_test - mu)**2)
        fold += 1
    meanError[i] = np.mean(error)
    i +=1
print meanError
bestParameter = kernalParameterRarnge[np.argmin(meanError)]
print "Best hyperparameter: ", bestParameter




kernalParameter = bestParameter
kernel = kernel_exp(normalizedData[::, 0].reshape(-1,1),np.reshape(normalizedData[::, 0], (-1, 1)), kernalParameter)


noise = np.identity(normalizedData.shape[0])*beta
cKernel = kernel + noise
L = np.linalg.cholesky(cKernel)

# Generate evenly spaced x axis for test
synTestData = np.linspace(0, 1, 100).reshape(-1,1)

# find mean from best hyperparameter
fac1 = np.linalg.solve(L, kernel_exp(normalizedData[::, 0].reshape(-1, 1), synTestData, kernalParameter))
fac2 = np.linalg.solve(L, normalizedData[::, 1])
mu = np.dot(np.transpose(fac1), fac2)
#mu = np.dot(np.transpose(kernel_sqexp), )


# Plot
plt.scatter(normalizedData[::, 0], normalizedData[::, 1])
plt.plot(synTestData, mu, 'r')
plt.title("Exponential Kernel Parameter")
plt.show()
