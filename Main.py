from math import *
import numpy as np

data = np.genfromtxt(fname='mnist_train.csv', delimiter=',')

data = np.delete(data, 0, 0)

data = data[0:100, :]

allLabels = data[:, 0]

data = np.delete(data, 0, 1)

training = data[0:75, :]
testing = data[75:-1, :]
trainingLabels = allLabels[0:75]
testingLabels = allLabels[75:-1]

def PointsDistance(preFeatures, currentPred):
    if len(preFeatures) == len(currentPred):
        distance = 0

        for i in range(len(preFeatures)):
            distance += (preFeatures[i] - currentPred[i])**2
        distance = sqrt(distance)

        return distance
    else:
        raise Exception("arrays are not the same size")

def Distance(features, currentPred):
    dists = []

    for i in features:
        dists.append(PointsDistance(i, currentPred))
    
    return dists


def AssignLabel(labels, dists, neighbors):
    minLabels = []
    labels2 = labels[:]

    for i in range(neighbors):
        minimum = min(dists)
        minIndex = dists.index(minimum)
        label = labels2[minIndex]
        minLabels.append(label)
        labels2 = np.delete(labels2, minIndex)
        del(dists[minIndex])

    newLabels = set(minLabels)

    return max(newLabels, key = minLabels.count)

def Predict(testing, training, trainingLabels, neighbors):
    predictedLabels = []

    for i in range(len(testing)):
        dists = Distance(training, testing[i])
        label = AssignLabel(trainingLabels, dists, neighbors)
        predictedLabels.append(label)

    return predictedLabels

def Score(testingLabels, predictedLabels):
    correct = 0

    for i in range(len(testingLabels)):
        if predictedLabels[i] == testingLabels[i]:
            correct += 1

    return correct / len(testingLabels)

predictions = Predict(testing, training, trainingLabels, 3)

print(Score(testingLabels, predictions))