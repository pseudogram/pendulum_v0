from deap import creator
from deap import base
from deap import tools
import numpy as np
import random

# Creates a Fitness class, with a weights attribute.
#    1 means a metric that needs to be maximized = the value
#    -1 means the metric needs to be minimized. All OpenAI
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Creates a class Individual, that is based on a numpy array as that is the
# class used for the weights passed to TensorFlow.
#   It also has an attribute that is a Fitness, when setting the attribute
#   it automatically calls the __init__() in Fitness initializing the
#   weight (1)
creator.create("Individual", list, fitness=creator.FitnessMax)
# ==============================================================================

toolbox = base.Toolbox()

# Create a function 'attr_item' to return the 'ID' of one item
# num_nodes = num_inputs+1 = 20
NUM_NODES = 20

# create a function 'individual'. It takes an individual and instantiates
#   it with a numpy array of size, NUM_NODES and type float32
toolbox.register("attr_floats", lambda n: np.random.rand(n).astype(np.float32)*0.1, NUM_NODES)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_floats)

# create a function 'population' that generates a list of individuals, x long.
#   NB: As repeat init takes 3 parameters, and as the first 2 have been given.
#       only one is needed.
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate( individual, MAX_REWARD=0 ):
    value = 0.0

    for item in individual:
        value += item  # Ensure overweighted bags are dominated
    return sum(individual),  # val


def cross( ind1, ind2 ):
    """Apply a crossover operation on input sets. The first child is the
    intersection of the two sets, the second child is the difference of the
    two sets.

    Is ind1 the winner or ind2???
    """
    for x in range(len(ind1)):
        if random() < 0.5:
            ind2[x] = ind1[x]  # Used in order to keep type
            # Symmetric Difference (inplace)
    return ind1, ind2

def mutate(individual):
    """Mutation that pops or add an element"""
    #     if random.random() < (1/len(individual)):

    # Select one gene
    index = random.randint(0,len(individual)-1)
    individual[index] *= (np.random.randn()*0.0025)
    return individual

toolbox.register("evaluate", evaluate)
toolbox.register("mate", cross)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=2)

import numpy
from deap import algorithms


def main():
    pop = toolbox.population(n=50)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 40

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

    return pop

if __name__ == "__main__":
    print(main())


