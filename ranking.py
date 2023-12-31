import random
import numpy as np
import torch
from collections import defaultdict

def getRandomScores(weightMatrix):
    networkScores = []
    maxScoreinLayer = []
    for index in range(1, len(weightMatrix.keys())):
        # for neuron_Num in range(1, weightMatrix['layers.'+str(2*index)+'.weight'].shape[1]+1):
        # layerScores.append(random.randint(1, 200))
        layerScores = list(range(1, weightMatrix['layers.' + str(2 * index) + '.weight'].shape[1] + 1))
        random.shuffle(layerScores)
        networkScores.append(layerScores)
        maxScoreinLayer.append(weightMatrix['layers.' + str(2 * index) + '.weight'].shape[1])

    lastLayer = [1] * weightMatrix['layers.' + str(2 * index) + '.weight'].shape[0]
    networkScores.append(lastLayer)
    maxScoreinLayer.append(1)

    return networkScores, maxScoreinLayer

def getRandomScoresVAE(weightMatrix):
    networkScores = []
    maxScoreinLayer = []
    what_i_need = ['encoder.5.weight', 'encoder.7.weight', 'fc_mu.weight',
                   'decoder.0.weight', 'decoder.2.weight', 'decoder.4.weight']
    for index in what_i_need:
        layerScores = list(range(1, weightMatrix[index].shape[0] + 1))
        random.shuffle(layerScores)
        networkScores.append(layerScores)
        if index == 'fc_mu.weight' or index == 'decoder.4.weight':
            maxScoreinLayer.append(2*weightMatrix[index].shape[0])
        else:
            maxScoreinLayer.append(weightMatrix[index].shape[0])

    return networkScores, maxScoreinLayer


def getLocalRanks(weightMatrix, activationMatrix):
    contribBasedPruning = True
    numOfHiddenLayers = 0
    for layerName in weightMatrix.keys():
        if 'fc_layers' in layerName:
            numOfHiddenLayers+=1

    #numOfHiddenLayers = len(weightMatrix.keys())
    #sampleSize = activationMatrix['layers.0.weight'].shape[0]
    networkScores = []
    maxScoreinLayer = []
    for layer in range(1, numOfHiddenLayers):
        currentWeights = weightMatrix['fc_layers.'+str(2*layer)+'.weight']
        currentActivations = activationMatrix['fc_layers['+str(2*(layer-1))+']']

        currentWeights = torch.abs(currentWeights)
        currentActivations = torch.abs(currentActivations)

        weightedActivations = np.dot(currentActivations, currentWeights.T)
        sampleSize = weightedActivations.shape[0]

        currLayerNumOfNeurons = currentWeights.shape[1]
        nextLayerNumOfNeurons = currentWeights.shape[0]
        avgActivationPerc = torch.zeros(currLayerNumOfNeurons)
        maxActivationPerc = torch.zeros(currLayerNumOfNeurons)

        #sampleSize = sorted(random.sample(range(1, 40000), 5000))
        for sample in range(sampleSize):
            #print(sample)
            for targetNeuron in range(nextLayerNumOfNeurons):
                totalActivationOfNeuron = weightedActivations[sample, targetNeuron]
                weight = currentWeights[targetNeuron]
                activation = currentActivations[sample]
                neuralContribution = (weight* activation*100)/totalActivationOfNeuron
                avgActivationPerc+= neuralContribution
                maxActivationPerc = torch.max(maxActivationPerc, neuralContribution)

        avgActivationPercTuples = []
        for neuron in range(len(avgActivationPerc)):
            avgActivation= avgActivationPerc[neuron]/ (nextLayerNumOfNeurons * sampleSize)
            maxActivation = maxActivationPerc[neuron]
            avgActivationPercTuples.append([neuron, avgActivation])
            #avgActivationPercTuples.append([neuron, (avgActivation + maxActivation)//2])

        sortedActivationPerc = sorted(avgActivationPercTuples, key=lambda x: x[1], reverse=contribBasedPruning)
        rank = 1
        for neuronInfo in sortedActivationPerc:
            neuronInfo.append(rank)
            rank+=1

        ranks = [0]* len(sortedActivationPerc)
        for i in range(len(sortedActivationPerc)):
            ranks[sortedActivationPerc[i][0]] = sortedActivationPerc[i][2]
        
        networkScores.append(ranks)
        maxScoreinLayer.append(len(sortedActivationPerc))

    lastLayer = [1] * weightMatrix['fc_layers.' + str(2 *(layer)) + '.weight'].shape[0]
    networkScores.append(lastLayer)
    maxScoreinLayer.append(1)

    return networkScores, maxScoreinLayer


def getRanksForVAE(weightMatrix, activationMatrix):
    numOfHiddenLayers = 0
    for layerName in weightMatrix.keys():
        if 'fc_layers' in layerName:
            numOfHiddenLayers+=1

    #numOfHiddenLayers = len(weightMatrix.keys())
    #sampleSize = activationMatrix['layers.0.weight'].shape[0]
    networkScores = []
    maxScoreinLayer = []
    for layer in range(1, numOfHiddenLayers):
        currentWeights = weightMatrix['fc_layers.'+str(2*layer)+'.weight']
        currentActivations = activationMatrix['fc_layers['+str(2*(layer-1))+']']

        currentWeights = torch.abs(currentWeights)
        currentActivations = torch.abs(currentActivations)

        weightedActivations = np.dot(currentActivations, currentWeights.T)
        sampleSize = weightedActivations.shape[0]

        currLayerNumOfNeurons = currentWeights.shape[1]
        nextLayerNumOfNeurons = currentWeights.shape[0]
        avgActivationPerc = torch.zeros(currLayerNumOfNeurons)
        maxActivationPerc = torch.zeros(currLayerNumOfNeurons)

        #sampleSize = sorted(random.sample(range(1, 40000), 5000))
        for sample in range(sampleSize):
            #print(sample)
            for targetNeuron in range(nextLayerNumOfNeurons):
                totalActivationOfNeuron = weightedActivations[sample, targetNeuron]
                weight = currentWeights[targetNeuron]
                activation = currentActivations[sample]
                neuralContribution = (weight* activation*100)/totalActivationOfNeuron
                avgActivationPerc+= neuralContribution
                maxActivationPerc = torch.max(maxActivationPerc, neuralContribution)

        avgActivationPercTuples = []
        for neuron in range(len(avgActivationPerc)):
            avgActivation= avgActivationPerc[neuron]/ (nextLayerNumOfNeurons * sampleSize)
            maxActivation = maxActivationPerc[neuron]
            avgActivationPercTuples.append([neuron, avgActivation])
            #avgActivationPercTuples.append([neuron, (avgActivation + maxActivation)//2])

        sortedActivationPerc = sorted(avgActivationPercTuples, key=lambda x: x[1], reverse=False)
        rank = 1
        for neuronInfo in sortedActivationPerc:
            neuronInfo.append(rank)
            rank+=1

        ranks = [0]* len(sortedActivationPerc)
        for i in range(len(sortedActivationPerc)):
            ranks[sortedActivationPerc[i][0]] = sortedActivationPerc[i][2]
        
        networkScores.append(ranks)
        maxScoreinLayer.append(len(sortedActivationPerc))

    lastLayer = [1] * weightMatrix['fc_layers.' + str(2 *(layer)) + '.weight'].shape[0]
    networkScores.append(lastLayer)
    maxScoreinLayer.append(1)

    return networkScores, maxScoreinLayer