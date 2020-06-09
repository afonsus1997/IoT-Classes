import operator
import random

import numpy as np
import math
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms, benchmarks
from mlp import runMLP

# random.seed(41)
maxlayers = 3
nbit = 8
maxneuron = 2**nbit # max 256 #must be a power of 2 because of the encoding

possible_features = ['Requests', 'Requests1','Requests2', 'Load', 'High_requests']
# possible_features_enum = {'Requests': 0, 'Requests1': 1,'Requests2': 2, 'Load': 3, 'High_requests': 4}

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def createInd():
    ind = []
    for i in range(5):
        ind.append(random.randint(0, 1)) #create bits for the features (ordered)
    for i in range(random.randint(1, maxlayers)): #create n layers
        for j in range(nbit):
            ind.append(random.randint(0, 1)) #create numer of neurons per layer
    return ind


def decodeIndividual(ind):
    outdict = {}
    outdict['layersizes'] = []
    outdict['input_features'] = []

    for i in range(5):
        if(ind[i]):
            outdict['input_features'].append(possible_features[i])

    ind_layers = ind[5:]
    layerslst = [ind_layers[i:i + 8] for i in range(0, len(ind_layers), 8)]
    for i in range(len(layerslst)):
        outdict['layersizes'].append(int(''.join(map(str, layerslst[i])), 2))

    return outdict

toolbox = base.Toolbox()



toolbox.register("indInitializer", createInd)
toolbox.register("individual", creator.Individual, toolbox.indInitializer())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# pop = toolbox.population(n=5)
# print(pop)
# for i in range(len(pop)):
#     print(decodeIndividual(pop[i]))
# quit()

def evalOptions(individual):
    individual_dict = decodeIndividual(individual)
    return (runMLP(tuple(individual_dict['layersizes']), individual_dict['input_features']),)



toolbox.register("evaluate", evalOptions)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, indpb=0.15)
toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("select", tools.selNSGA2)

# ind1 = toolbox.individual()
# ind2 = toolbox.individual()


pop = toolbox.population(n=10)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, stats=stats, halloffame=hof, verbose=True)
gen, avg, min_, max_ = logbook.select("gen", "avg", "min", "max")
plt.plot(gen, avg, label="average")
plt.plot(gen, min_, label="minimum")
plt.plot(gen, max_, label="maximum")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend(loc="lower right")
# plt.show()
plt.savefig('results.png')

# ind = toolbox.individual()
# print(ind)
# ind = toolbox.mutate(ind)
# print(ind)