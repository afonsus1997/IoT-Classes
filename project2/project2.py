import operator
import random

import numpy
import math
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms, benchmarks
from mlp import runMLP

# random.seed(41)
maxlayers = 3
maxneuron = 200

possible_features = ['Requests', 'Requests1','Requests2', 'Load', 'High_requests']
# possible_features_enum = {'Requests': 0, 'Requests1': 1,'Requests2': 2, 'Load': 3, 'High_requests': 4}


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", dict, fitness=creator.FitnessMax)

# featureenumlist = ['number_of_requests', 'number_of_requests1', 'number_of_requests']

def createAndEncodeIndividual():
    outdict = {}
    outdict['layersizes'] = []
    outdict['input_features'] = []

    for i in range(random.randint(1, maxlayers)):
        outdict['layersizes'].append(random.randint(1, maxneuron))

    tmpfeatures = set(possible_features)
    outdict['input_features'] = random.sample(tmpfeatures, random.randint(1, 5))

    #other parameters to be encoded here

    return outdict




toolbox = base.Toolbox()
# toolbox.register("rng_layers", random.randint, 1, maxneuron)
# toolbox.register("rng_nlayers", random.randint, 1, maxlayers)
# toolbox.register("create_layers", tools.initRepeat, list, toolbox.rng_layers, n=toolbox.rng_nlayers())
# toolbox.register("indInitializer", createAndEncodeIndividual, toolbox.create_layers())
# toolbox.register("individual", creator.Individual, toolbox.indInitializer())
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)


toolbox.register("indInitializer", createAndEncodeIndividual)
toolbox.register("individual", creator.Individual, toolbox.indInitializer())
toolbox.register("population", tools.initRepeat, list, toolbox.individual)



def evalOptions(individual):
    return (runMLP(tuple(individual['layersizes']), individual['input_features']),)

def cxInds(ind1, ind2):
    ind1_tmp = ind1['input_features']
    ind2_tmp = ind2['input_features']

    indslayers = set({tuple(ind1['layersizes']), tuple(ind2['layersizes'])})

    ind1['input_features'] = [value for value in ind1_tmp if value in ind2_tmp]
    ind2['input_features'] = [value for value in ind2_tmp if value in ind1_tmp]
    # ind2['input_features'] = [value for value in ind1_tmp if value not in ind2_tmp] + [value for value in ind2_tmp if value not in ind1_tmp]

    ind1['layersizes'] = list(random.sample(indslayers, 1)[0])
    ind2['layersizes'] =  list(random.sample(indslayers, 1)[0])

    return ind1, ind2

def mutInd(indin):
    #possible mutations: remove/add feature randomize layers?

    number_of_layers = len(indin['layersizes'])
    possible_layers = set(range(0, number_of_layers))
    # print(possible_layers)
    layers_to_change = random.sample(possible_layers, random.randint(1, len(possible_layers)))
    # print(layers_to_change)
    for i in range(len(layers_to_change)):
        indin['layersizes'][layers_to_change[i]] = random.randint(1, maxneuron)

    return indin,


toolbox.register("evaluate", evalOptions)
toolbox.register("mate", cxInds)
toolbox.register("mutate", mutInd)
toolbox.register("select", tools.selNSGA2)
# toolbox.register("select", tools.selTournament, tournsize=3)

# ind1 = toolbox.individual()
# ind2 = toolbox.individual()
#
# print(cxInds(ind1, ind2))


pop = toolbox.population(n=10)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean)
stats.register("min", numpy.min)
stats.register("max", numpy.max)

pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, stats=stats, halloffame=hof, verbose=True)
gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")
plt.plot(gen, avg, label="average")
plt.plot(gen, min_, label="minimum")
plt.plot(gen, max_, label="maximum")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend(loc="lower right")
plt.show()


# ind = toolbox.individual()
# print(ind)
# ind = toolbox.mutate(ind)
# print(ind)