import random


def getRandomScores(weightMatrix):
    networkScores = []
    maxScoreinLayer = []
    for index in range(1, len(weightMatrix.keys())):
        #for neuron_Num in range(1, weightMatrix['layers.'+str(2*index)+'.weight'].shape[1]+1):
            #layerScores.append(random.randint(1, 200))
        layerScores = list(range(1, weightMatrix['layers.'+str(2*index)+'.weight'].shape[1]+1))
        random.shuffle(layerScores)
        networkScores.append(layerScores)
        maxScoreinLayer.append(weightMatrix['layers.'+str(2*index)+'.weight'].shape[1])

    lastLayer =  [1]* weightMatrix['layers.'+str(2*index)+'.weight'].shape[0]
    networkScores.append(lastLayer)
    maxScoreinLayer.append(1)
    
    return networkScores, maxScoreinLayer