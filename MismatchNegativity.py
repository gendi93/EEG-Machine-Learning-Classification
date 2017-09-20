'''
Author: Yehia El Gendi
Date: 19/08/2017
'''

import scipy.io as sio
from scipy.fftpack import rfft
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# store the responses in arrays
def storeResponses(experiment, participant):
    data = experiment[0,participant]
    standardsFZ = []
    standardsCZ = []
    standardsFCZ = []
    standardsVEOG = []
    standardsHEOG = []
    deviantsFZ = []
    deviantsCZ = []
    deviantsFCZ = []
    deviantsVEOG = []
    deviantsHEOG = []
    for tuple in standardDurations:
        standardsFZ.append(data['data'][np.where(data['electrode']['labels']=='FZ')[1].item(0),tuple[0].item(0):tuple[0].item(0)+tuple[1].item(0)].tolist())
        standardsCZ.append(data['data'][np.where(data['electrode']['labels']=='CZ')[1].item(0),tuple[0].item(0):tuple[0].item(0)+tuple[1].item(0)].tolist())
        standardsFCZ.append(data['data'][np.where(data['electrode']['labels']=='FCZ')[1].item(0),tuple[0].item(0):tuple[0].item(0)+tuple[1].item(0)].tolist())
        standardsVEOG.append(data['data'][np.where(data['electrode']['labels']=='VEOG')[1].item(0),tuple[0].item(0):tuple[0].item(0)+tuple[1].item(0)].tolist())
        standardsHEOG.append(data['data'][np.where(data['electrode']['labels']=='HEOG')[1].item(0),tuple[0].item(0):tuple[0].item(0)+tuple[1].item(0)].tolist())
    for tuple in deviantDurations:
        deviantsFZ.append(data['data'][np.where(data['electrode']['labels']=='FZ')[1].item(0),tuple[0].item(0):tuple[0].item(0)+tuple[1].item(0)].tolist())
        deviantsCZ.append(data['data'][np.where(data['electrode']['labels']=='CZ')[1].item(0),tuple[0].item(0):tuple[0].item(0)+tuple[1].item(0)].tolist())
        deviantsFCZ.append(data['data'][np.where(data['electrode']['labels']=='FCZ')[1].item(0),tuple[0].item(0):tuple[0].item(0)+tuple[1].item(0)].tolist())
        deviantsVEOG.append(data['data'][np.where(data['electrode']['labels']=='VEOG')[1].item(0),tuple[0].item(0):tuple[0].item(0)+tuple[1].item(0)].tolist())
        deviantsHEOG.append(data['data'][np.where(data['electrode']['labels']=='HEOG')[1].item(0),tuple[0].item(0):tuple[0].item(0)+tuple[1].item(0)].tolist())
    return standardsFZ, standardsCZ, standardsFCZ, standardsVEOG, standardsHEOG, deviantsFZ, deviantsCZ, deviantsFCZ, deviantsVEOG, deviantsHEOG
# calculate the durations of all responses
def calculateDurations(experiment, participant):
    data = experiment[0,participant]
    standardDurations = []
    deviantDurations = []
    for event in range(0,data['event'].shape[1]-1):
        if data['event']['type'][0,event] == 101:
            standardDurations.append([data['event']['latency'][0,event], data['event']['latency'][0,event+1]-data['event']['latency'][0,event]])
        elif data['event']['type'][0,event] == 102:
            deviantDurations.append([data['event']['latency'][0,event+1], data['event']['latency'][0,event+1]-data['event']['latency'][0,event]])
    return standardDurations, deviantDurations
# get the number of responses
def getNumResponses(standards, deviants):
    return len(standards[0]), len(deviants[0])
# get the shortest response time
def getShortestResponse(experiment):
    shortestOverallStandard = 400
    shortestOverallDeviant = 400
    for participant in range(0, len(experiment[0])):
        shortestStandard = 400
        shortestDeviant = 400
        standardDurations, deviantDurations = calculateDurations(experiment, participant)
        for tuple in standardDurations:
            if tuple[1] < shortestStandard:
                shortestStandard = tuple[1].item(0)
        for tuple in deviantDurations:
            if tuple[1] < shortestDeviant:
                shortestDeviant = tuple[1].item(0)
        shortestOverallStandard = (shortestStandard if shortestStandard < shortestOverallStandard else shortestOverallStandard)
        shortestOverallDeviant = (shortestDeviant if shortestDeviant < shortestOverallDeviant else shortestOverallDeviant)
    return shortestOverallStandard, shortestOverallDeviant
# combines the responses from the specified electrodes into a single array
def combineElectrodes(standardsFZ, standardsCZ, standardsFCZ, standardsVEOG, standardsHEOG, deviantsFZ, deviantsCZ, deviantsFCZ, deviantsVEOG, deviantsHEOG):
    numStandards, numDeviants = getNumResponses(standardsFZ, deviantsFZ)
    standards = []
    deviants = []
    standards.append(standardsFZ)
    standards.append(standardsCZ)
    standards.append(standardsFCZ)
    standards.append(standardsVEOG)
    standards.append(standardsHEOG)
    deviants.append(deviantsFZ)
    deviants.append(deviantsCZ)
    deviants.append(deviantsFCZ)
    deviants.append(deviantsVEOG)
    deviants.append(deviantsHEOG)
    return standards, deviants
# trims all the responses to the shortest response
def trimResponses(standards, deviants, shortestOverallStandard, shortestOverallDeviant):
    trimmedStandards = []
    trimmedDeviants = []
    electrodeIndex = 0
    responseIndex = 0
    for electrode in standards:
        trimmedStandards.append([])
        responseIndex = 0
        for response in electrode:
            trimmedStandards[electrodeIndex].append(response[0:shortestOverallStandard])
            responseIndex+=1
        electrodeIndex+=1
    electrodeIndex = 0
    for electrode in deviants:
        trimmedDeviants.append([])
        responseIndex = 0
        for response in electrode:
            trimmedDeviants[electrodeIndex].append(response[0:shortestOverallDeviant])
            responseIndex += 1
        electrodeIndex += 1
    return trimmedStandards, trimmedDeviants
# averages all the responses
def averageResponses(trimmedStandards, trimmedDeviants, numStandards, numDeviants, shortestStandard, shortestDeviant):
    averageStandard = []
    averageDeviant = []
    electrodeIndex = 0
    for electrode in trimmedStandards:
        averageStandard.append([0]*shortestStandard)
        for response in electrode:
            averageStandard[electrodeIndex] = [sum(x) for x in zip(averageStandard[electrodeIndex], response)]
        averageStandard[electrodeIndex] = [x/numStandards for x in averageStandard[electrodeIndex]]
        electrodeIndex += 1
    electrodeIndex = 0
    for electrode in trimmedDeviants:
        averageDeviant.append([0]*shortestDeviant)
        for response in electrode:
            averageDeviant[electrodeIndex] = [sum(x) for x in zip(averageDeviant[electrodeIndex], response)]
        averageDeviant[electrodeIndex] = [x/numDeviants for x in averageDeviant[electrodeIndex]]
        electrodeIndex += 1
    return averageStandard, averageDeviant
# adds an offset to the average response to bring it to zero level
def zeroResponses(averageStandard, averageDeviant):
    zeroStandard = []
    zeroDeviant = []
    for electrode in averageStandard:
        zeroStandard.append([])
        offset = electrode[0]
        zeroStandard[averageStandard.index(electrode)] = [x - offset for x in electrode]
    for electrode in averageDeviant:
        zeroDeviant.append([])
        offset = electrode[0]
        zeroDeviant[averageDeviant.index(electrode)] = [x - offset for x in electrode]
    return zeroStandard, zeroDeviant
# calculates the fourier transform frequencies and amplitudes for responses
def fourierTransformResponses(standards, deviants, shortestOverallStandard, shortestOverallDeviant):
    fourierResponses = []
    types = []
    clip = (shortestOverallDeviant if shortestOverallDeviant < shortestOverallStandard else shortestOverallStandard)
    for electrode in range(0,3):
        for response in standards[electrode]:
            amplitudes = rfft(response)
            fourierResponses.append(abs(amplitudes[0:clip]).tolist())
            types.append('standard')
        for response in deviants[electrode]:
            amplitudes = rfft(response)
            fourierResponses.append(abs(amplitudes[0:clip]).tolist())
            types.append('deviant')
    return fourierResponses, types
# calculate the mismatch negaivity from a standard and deviant response
def calculateMMN(zeroStandard, zeroDeviant):
    return [x-zeroStandard[2][zeroDeviant[2].index(x)] for x in zeroDeviant[2][0:300]]
# plot the average electrode responses
def plotAverageResponses(zeroStandard, zeroDeviant):
    electrodeNames = ['Fz','Cz','FCz','VEOG','HEOG']
    fig = plt.figure()
    for i in range(0,5):
        a=fig.add_subplot(2,3,i+1)
        plt.plot(np.arange(0,250)*2, zeroStandard[i][60:310],'red')
        plt.plot(np.arange(0,250)*2, zeroDeviant[i][60:310],'blue')
        a.set_xlabel('Latency (ms)')
        a.set_ylabel('Potential (microVolts)')
        a.set_title(electrodeNames[i])
        plt.grid()
    plt.show()
# plot the frequency domain of all responses
def plotFrequencyAmplitudes(fourierResponses,types):
    reds = [x for x in fourierResponses if types[fourierResponses.index(x)] == 'standard']
    blues = [x for x in fourierResponses if types[fourierResponses.index(x)] == 'deviant']
    for red in reds:
        plt.plot(red,'red')
    for blue in blues:
        plt.plot(blue,'blue')
    plt.show()
# find best fit for a specific classifier
def classify(classifier, parameters):
    learnLimit = parameters[0]
    testLabels = TYPES[learnLimit:len(FOURIERS)-1]
    numDevs = sum([x=='deviant' for x in testLabels])
    numStds = sum([x=='standard' for x in testLabels])
    if classifier == "Logistic Regression" or classifier == "Linear SVM" or classifier == "Nearest Neighbours":
        CList = parameters[1]
        neighbours = parameters[2]
        if classifier == "Logistic Regression":
            parameterGrid = dict(C=CList)
            clf = GridSearchCV(LogisticRegression(), param_grid=parameterGrid, cv=5, verbose=10, refit=True)
        elif classifier == "Linear SVM":
            parameterGrid = dict(C=CList)
            clf = GridSearchCV(svm.LinearSVC(), param_grid=parameterGrid, cv=5, verbose=10, refit=True)
        elif classifier == "Nearest Neighbours":
            parameterGrid = dict(n_neighbors=neighbours)
            clf = GridSearchCV(neighbors.KNeighborsClassifier(), param_grid=parameterGrid, cv=5, verbose=10, refit=True)
        clfFit = clf.fit(FOURIERS[0:learnLimit], TYPES[0:learnLimit])
        testPredictions = clfFit.predict(FOURIERS[learnLimit:len(FOURIERS)-1])
        print("\nThe best mean accuracy is: ", clfFit.best_score_)
        print("Best parameters: ", clfFit.best_params_)
        confusionMatrix = confusion_matrix(testLabels, testPredictions)
        standardErrorRate = confusionMatrix[1,0]*100/numStds
        deviantErrorRate = confusionMatrix[0,1]*100/numDevs
        accuracy = (confusionMatrix[0,0]+confusionMatrix[1,1])/(confusionMatrix[0,0]+confusionMatrix[0,1]+confusionMatrix[1,0]+confusionMatrix[1,1])
        print("Best accuracy: {:0.3f}".format(accuracy))
        print(confusionMatrix)
        print("standard error rate = {:0.2f}%".format(standardErrorRate))
        print("deviant error rate = {:0.2f}%\n".format(deviantErrorRate))
        print(classification_report(testLabels, testPredictions))
    elif classifier == "Decision Trees":
        accuracies = []
        standardErrorRates = []
        deviantErrorRates = []
        clf = DecisionTreeClassifier()
        for i in range(0,10):
            print("Iteration: {}".format(i+1))
            testPredictions = predict(clf, FOURIERS, TYPES, learnLimit)
            accuracies, standardErrorRates, deviantErrorRates = reportAccuracy(accuracies, testPredictions, testLabels, standardErrorRates, deviantErrorRates, numStds, numDevs)
        bestAccuracy = max(accuracies)
        bestStandardErrorRate = standardErrorRates[accuracies.index(bestAccuracy)]
        bestDeviantErrorRate = deviantErrorRates[accuracies.index(bestAccuracy)]
        averageAccuracy = np.mean(np.array(accuracies))
        confusionMatrix = confusion_matrix(testLabels, testPredictions)
        standardErrorRate = confusionMatrix[1,0]*100/numStds
        deviantErrorRate = confusionMatrix[0,1]*100/numDevs
        accuracy = (confusionMatrix[0,0]+confusionMatrix[1,1])/(confusionMatrix[0,0]+confusionMatrix[0,1]+confusionMatrix[1,0]+confusionMatrix[1,1])
        print("Best accuracy: {:0.3f}".format(accuracy))
        print(confusionMatrix)
        print("standard error rate = {:0.2f}%".format(standardErrorRate))
        print("deviant error rate = {:0.2f}%\n".format(deviantErrorRate))
        print(classification_report(testLabels, testPredictions))
        print("average accuracy = {:0.4f}".format(averageAccuracy*100))
        print("best accuracy: {:0.4f}%, standard error: {:0.4f}%, deviant error: {:0.4f}%\n\n\n".format(bestAccuracy*100, bestStandardErrorRate, bestDeviantErrorRate))
# make predictions based on the current classifier
def predict(clf, FOURIERS, TYPES, learnLimit):
    clf.fit(FOURIERS[0:learnLimit], TYPES[0:learnLimit])
    testPredictions = clf.predict(FOURIERS[learnLimit:len(FOURIERS)-1])
    return testPredictions
# print results of selected classifier and parameters
def reportAccuracy(accuracies, testPredictions, testLabels, standardErrorRates, deviantErrorRates, numStds, numDevs):
    confusionMatrix = confusion_matrix(testLabels, testPredictions)
    accuracy = accuracy_score(testLabels, testPredictions)
    standardErrorRate = confusionMatrix[1,0]*100/numStds
    deviantErrorRate = confusionMatrix[0,1]*100/numDevs
    accuracies.append(accuracy)
    standardErrorRates.append(standardErrorRate)
    deviantErrorRates.append(deviantErrorRate)
    print("accuracy = {:0.2f}%".format(accuracy*100))
    print("standard error rate = {:0.2f}%".format(standardErrorRate))
    print("deviant error rate = {:0.2f}%\n".format(deviantErrorRate))
    return accuracies, standardErrorRates, deviantErrorRates
# import data from MATLAB

BSS = sio.loadmat('BSS.mat')
A = BSS['A_bss']
B = BSS['B_bss']
C = BSS['C_bss']
D = BSS['D_bss']
E = BSS['E_bss']
F = BSS['F_bss']
G = BSS['G_bss']
H = BSS['H_bss']
I = BSS['I_bss']
J = BSS['J_bss']
K = BSS['K_bss']
L = BSS['L_bss']
M = BSS['M_bss']
N = BSS['N_bss']
O = BSS['O_bss']
experiments = [[A,'A'], [B,'B'], [C,'C'], [D,'D'], [E,'E'], [F,'F'], [G,'G'], [H,'H'], [I,'I'], [J,'J'], [K,'K'], [L,'L'], [M,'M'], [N,'N'], [O,'O']]

N1Latencies = []
P1Latencies = []
MMNLatencies = []
for experiment in experiments:
    print("Experiment: {}\n".format(experiment[1]))
    FOURIERS = []
    TYPES = []
    shortestOverallStandard, shortestOverallDeviant = getShortestResponse(experiment[0])
    for participant in range(0, len(experiment[0][0])):
        print("Participant: {}\n".format(participant+1))
        standardDurations, deviantDurations = calculateDurations(experiment[0], participant)
        standardsFZ, standardsCZ, standardsFCZ, standardsVEOG, standardsHEOG, deviantsFZ, deviantsCZ, deviantsFCZ, deviantsVEOG, deviantsHEOG = storeResponses(experiment[0], participant)
        standards, deviants = combineElectrodes(standardsFZ, standardsCZ, standardsFCZ, standardsVEOG, standardsHEOG, deviantsFZ, deviantsCZ, deviantsFCZ, deviantsVEOG, deviantsHEOG)
        fourierResponses, types = fourierTransformResponses(standards, deviants, shortestOverallDeviant, shortestOverallStandard)
        FOURIERS = FOURIERS + fourierResponses
        TYPES = TYPES + types
        trimmedStandards, trimmedDeviants = trimResponses(standards, deviants, shortestOverallStandard, shortestOverallDeviant)
        numStandards, numDeviants = getNumResponses(standards, deviants)
        averageStandard, averageDeviant = averageResponses(trimmedStandards, trimmedDeviants, numStandards, numDeviants, shortestOverallStandard, shortestOverallDeviant)
        zeroStandard, zeroDeviant = zeroResponses(averageStandard, averageDeviant)
        MMN = calculateMMN(averageStandard, averageDeviant)
        MMNLatency = (np.arange(0,250)*2).tolist()[MMN.index(min(MMN[100:175]))]-100
        MMNLatencies.append(MMNLatency)
        plotAverageResponses(zeroStandard, zeroDeviant)
    learnLimit = len(FOURIERS)*9//10 # controls the training/testing split
    CList = 10. ** np.arange(-5,10) # logarithmic range of values for C
    neighbours = [1,2,3,4]
    parameters = [learnLimit, CList, neighbours]
    # Substitute the string below for the desired classifier: "Logistic Regression", "Linear SVM", "RBF SVM", "Decision Trees" or "Nearest Neighbours"
    classify("Logistic Regression", parameters)
