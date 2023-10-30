import random


def getRandomScores(weightMatrix):
    scoreCard = {}
    for index in range(1, len(weightMatrix.keys())):
        for neuron_Num in range(1, weightMatrix['weights['+str(2*index)+']'].shape[1]+1):
            scoreCard['HiddenLayer'+str(index)+'_'+'Neuron'+str(neuron_Num)] = random.randint(1, 200)
    
    return scoreCard