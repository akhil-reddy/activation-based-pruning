import random


def getRandomScores(weightMatrix):
    networkScores = []
    for index in range(1, len(weightMatrix.keys())):
        layerScores = []
        for neuron_Num in range(1, weightMatrix['weights['+str(2*index)+']'].shape[1]+1):
            layerScores.append(random.randint(1, 200))
        networkScores.append(layerScores)
    
    return networkScores