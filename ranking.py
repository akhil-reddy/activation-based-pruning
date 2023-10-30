import random

def getRandomScores(weightMatrix):
    networkScores = []
    for index in range(1, len(weightMatrix.keys())):
        layerScores = []
        for neuron_Num in range(1, weightMatrix['layers.'+str(2*index)+'.weight'].shape[1]+1):
            layerScores.append(random.randint(1, 200))
        networkScores.append(layerScores)
    
    return networkScores

""" 
            Normal  Special
    Local 
    Global
"""