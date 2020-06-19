import operator
import random

import numpy as np
import math
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms, benchmarks
from mlp import runMLP
from fixed_crossover import*
from fixed_mutation import*

hofind = []
hofit = []

# random.seed(41)
maxlayers = 3
nbit = 8
maxneuron = 2**nbit # max 256 #must be a power of 2 because of the encoding

possible_features = ['Requests', 'Requests1','Requests2', 'Load', 'High_requests', 'High_load', 'High_features', 'Requests_mm', 'Load_mm']

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("rng_bit", random.randint, 0, 1)
toolbox.register("rng_nlayers", random.randint, 1, maxlayers)

def decodeIndividual(ind):
    outdict = {}
    outdict['layersizes'] = []
    outdict['input_features'] = []

    for i in range(9):
        if(ind[i]):
            outdict['input_features'].append(possible_features[i])

    ind_layers = ind[9:]
    layerslst = [ind_layers[i:i + 8] for i in range(0, len(ind_layers), 8)]
    for i in range(len(layerslst)):
        outdict['layersizes'].append(int(''.join(map(str, layerslst[i])), 2))

    return outdict

def checkInd(ind):
    inddic = decodeIndividual(ind)
    if(inddic["input_features"] == []):
        return False
    elif 0 in set(inddic['layersizes']):
        return False
    else:
        return True


def createInd():
    while(True):
        ind = []
        for i in range(9):
            ind.append(toolbox.rng_bit()) #create bits for the features (ordered)
        for i in range(toolbox.rng_nlayers()): #create n layers
            for j in range(nbit):
                ind.append(toolbox.rng_bit()) #create numer of neurons per layer
        if checkInd(ind):
            break
    return ind





toolbox.register("indInitializer", createInd)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.indInitializer, 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOptions(individual):
    individual_dict = decodeIndividual(individual[0])
    if(checkInd(individual[0]) == False):
        return (0,)
    else:
        fitness = runMLP(tuple(individual_dict['layersizes']), individual_dict['input_features'], False)
        hofind.append(individual_dict)
        hofit.append(fitness)
        return (fitness,)


toolbox.register("evaluate", evalOptions)
toolbox.register("mate", cxTwoPointfix)
toolbox.register("mutate", mutFlipBitfix, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


pop = toolbox.population(n=50)
for i in range(len(pop)):
    print(decodeIndividual(pop[i][0]))

hof = tools.HallOfFame(5)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, stats=stats, halloffame=hof, verbose=True)
gen, avg, min_, max_ = logbook.select("gen", "avg", "min", "max")


# print(hof)
# for i in range(len(hof)):
#     print("number " + str(i) + ":")
#     print(decodeIndividual(hof[i][0]))

plt.plot(gen, avg, label="average")
plt.plot(gen, min_, label="minimum")
plt.plot(gen, max_, label="maximum")
plt.xlabel("Generation")
# plt.show()
plt.ylabel("Fitness")
plt.legend(loc="lower right")
plt.savefig('results.png')

#OLD HOF
val_dicts = []
for i in range(len(hof)):
    val_dicts[i] = decodeIndividual(hof[i][0])
    fval = runMLP(tuple(val_dicts[i]['layersizes']), val_dicts[i]['input_features'], True)

# NEW HOF
# maxvalues = max(hofit) # this is not a great solution, but ensures that i get the best ind, instead of the hof method
# bestinds = [i for i, j in enumerate(hofit) if j == maxvalues]
# print("Best inds:")
# print(bestinds)
# for i in range(len(bestinds)):
#     print("=====================================")
#     print("Test Fitness "+ str(i) + " out of " + str(len(bestinds)-1) + ": " + str(hofit[bestinds[i]]))
#     fval = runMLP(tuple(hofind[bestinds[i]]['layersizes']), hofind[bestinds[i]]['input_features'], True)
#
