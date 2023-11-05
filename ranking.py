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


def getLocalRanks(weightMatrix, activationMatrix):
    numOfHiddenLayers = len(weightMatrix.keys())-1
    #sampleSize = activationMatrix['layers.0.weight'].shape[0]
    networkScores = []
    maxScoreinLayer = []
    for layer in range(1, numOfHiddenLayers+1):
        currentWeights = weightMatrix['layers.'+str(2*layer)+'.weight']
        currentActivations = activationMatrix['layers['+str(2*(layer-1))+']']

        currentWeights = torch.abs(currentWeights)
        currentActivations = torch.abs(currentActivations)

        weightedActivations = np.dot(currentActivations, currentWeights.T)
        sampleSize = weightedActivations.shape[0]

        currLayerNumOfNeurons = currentWeights.shape[1]
        nextLayerNumOfNeurons = currentWeights.shape[0]
        avgActivationPerc = defaultdict(lambda: 0)
        maxActivationPerc = defaultdict(lambda: 0)

        sampleSize = random.sample(range(1, 40000), 1000)
        for sample in sampleSize:
            print(sample)
            for targetNeuron in range(nextLayerNumOfNeurons):
                totalActivationOfNeuron = weightedActivations[sample, targetNeuron]
                for srcNeuron in range(currLayerNumOfNeurons):
                    weight = currentWeights[targetNeuron, srcNeuron]
                    activation = currentActivations[sample, srcNeuron]
                    neuralContribution = (abs(weight* activation)*100)/totalActivationOfNeuron
                    avgActivationPerc[srcNeuron]+= neuralContribution
                    maxActivationPerc[srcNeuron] = max(maxActivationPerc[srcNeuron], neuralContribution)
        avgActivationPercTuples = []
        for neuron in avgActivationPerc.keys():
            avgActivation= avgActivationPerc[neuron]/ (nextLayerNumOfNeurons * len(sampleSize))
            avgActivationPercTuples.append([neuron, avgActivation])

        sortedActivationPerc = sorted(avgActivationPercTuples, key=lambda x: x[1])
        rank = 1
        for neuronInfo in sortedActivationPerc:
            neuronInfo.append(rank)
            rank+=1

        ranks = [0]* len(sortedActivationPerc)
        for i in range(len(sortedActivationPerc)):
            ranks[sortedActivationPerc[i][0]] = sortedActivationPerc[i][2]
        
        networkScores.append(ranks)
        maxScoreinLayer.append(len(sortedActivationPerc))

    lastLayer = [1] * weightMatrix['layers.' + str(2 *(layer+1)) + '.weight'].shape[0]
    networkScores.append(lastLayer)
    maxScoreinLayer.append(1)

    return networkScores, maxScoreinLayer
