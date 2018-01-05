import numpy as np
from random import randint
import matplotlib.pyplot as plt
transMat = np.array([[0.95, 0.05],[0.10, 0.90]])
emissionMat = np.array(([[0.166, 0.166, 0.166, 0.166, 0.166, 0.166], [ 0.1,  0.1, 0.1, 0.1, 0.1, 0.5]]))
#steps = 100
#initialStateProb = np.array([1, 0])

#stepProb = np.zeros((2, steps))
#newInitial = np.dot(transMat, initialStateProb)
noSteps = 500
noStates = 2
u = np.random.uniform(0, 1, noSteps)
z = np.zeros((noSteps))
z[0] = 0 # Initial step
for i in ((np.arange(noSteps))-1):
    if z[i -1] == 0:
        z[i] = 0 if u[i] <= 0.95 else 1
    else:
        z[i] = 1 if u[i] <= 0.90 else 0
    i += 1


# Generating

x = np.zeros((noSteps))
for i in np.arange(noSteps):
    x[i] = randint(1, 6) if z[i] == 0 else np.random.choice(np.arange(1, 7), p = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5])

forward = np.zeros((noStates,noSteps + 1), float)
#forward[:, 0] = [1, 0] # fair has probability 1
forward[:, 0] = [1, 0]
for obsNumber in np.arange(noSteps):
    temp = np.matrix(forward[:, obsNumber])
    forward[:, obsNumber+1] = temp * np.matrix(transMat) * np.matrix(np.diag(emissionMat[:,(x[obsNumber]-1)]))
    forward[:, obsNumber + 1] = forward[:, obsNumber + 1] / np.sum(forward[:, obsNumber + 1])
backward = np.zeros((noStates,noSteps + 1))
backward[:,-1] = 1.0
for obsNumberRev in np.arange(noSteps, 0, -1):
    temp = np.matrix(backward[:, obsNumberRev]).transpose()
    backward[:, obsNumberRev - 1] = (np.matrix(transMat) * np.matrix(np.diag(emissionMat[:, x[obsNumberRev - 1] -
                                                                                            1])) *  temp).transpose()
    backward[:, obsNumberRev - 1] = backward[:, obsNumberRev - 1] / np.sum(backward[:, obsNumberRev - 1])

probMat = np.array(forward)*np.array(backward)
probMat = probMat/np.sum(probMat, 0)



plt.plot(np.arange(noSteps + 1), forward[1, ::])
plt.plot(np.arange(noSteps), z)
plt.ylim([-0.01, 1.01])
plt.show()

plt.plot(np.arange(noSteps + 1), probMat[1, ::])
plt.plot(np.arange(noSteps), z)
plt.ylim([-0.01, 1.01])
plt.show()
print "Yahoo!!!"

