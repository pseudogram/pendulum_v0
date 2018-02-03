from deap import creator
from deap import base
from deap import tools
import numpy as np
import random
import gym
from rnn import Basic_rnn, FullyConnectedRNN
from pprint import pprint

# ------------------------------------------------------------------------------
#                        SET UP: OpenAI Gym Environment
# ------------------------------------------------------------------------------

ENV_NAME = 'Pendulum-v0'
EPISODES = 3  # Number of times to run envionrment when evaluating
STEPS = 200  # Max number of steps to run run simulation

env = gym.make(ENV_NAME)

# Used to create controller
obs_dim = env.observation_space.shape[0]  # Input to controller (observ.)
action_dim = env.action_space.shape[0]  # Output from controller (action)
nodes = 10  # Unconnected nodes in network in the
# dt = 0.05  # dt for the environment, found in environment source code
dt = 1 # Works the best, 0.05 causes it to vanish

# ------------------------------------------------------------------------------
#                          SET UP: TensorFlow Basic_rnn
# ------------------------------------------------------------------------------
# agent = Basic_rnn(obs_dim, action_dim, nodes, dt, False)
agent = FullyConnectedRNN(obs_dim, action_dim, nodes)

# ------------------------------------------------------------------------------
#                               SET UP GA PARAMETERS
# ------------------------------------------------------------------------------
POPULATION_SIZE = 40
CROSS_PROB = 0.5
NUM_GEN = 2000   # Number of generations
DEME_SIZE = 3  # from either side

# ------------------------------------------------------------------------------
#                               CREATE GA
# ------------------------------------------------------------------------------

# Creates a Fitness class, with a weights attribute.
#    1 means a metric that needs to be maximized = the value
#    -1 means the metric needs to be minimized. All OpenAI
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Creates a class Individual, that is based on a numpy array as that is the
# class used for the weights passed to TensorFlow.
#   It also has an attribute that is a Fitness, when setting the attribute
#   it automatically calls the __init__() in Fitness initializing the
#   weight (1)
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
# ==============================================================================

toolbox = base.Toolbox()

# Create a function 'attr_item' to return the 'ID' of one item
# num_nodes = num_inputs+1 = 20
NUM_PARAMS = agent.num_params

# create a function 'individual'. It takes an individual and instantiates
#   it with a numpy array of size, NUM_PARAMS and type float32
# Set the weights to a value n where, -0.1 < n < 0.1
toolbox.register("attr_floats", lambda n: (np.random.rand(n).astype(
    np.float32)-0.5)*2, NUM_PARAMS)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_floats)

# create a function 'population' that generates a list of individuals, x long.
#   NB: As repeat init takes 3 parameters, and as the first 2 have been given.
#       only one is needed.
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# ------------------------------------------------------------------------------
#                         Evaluation function of GA
# ------------------------------------------------------------------------------
def evaluate(individual, MAX_REWARD=0 ):
    """Lends heavily from evaluate.py"""
    # Load weights into RNN
    agent.set_weights(individual)

    total_reward = 0
    for episode in range(EPISODES):
        # print("Starting Episode: {}".format(episode))

        # Starting observation
        observation = env.reset()

        episode_reward = 0
        next_state = np.random.rand(agent.state_size, 1).astype(
            dtype=np.float32)
        for step in range(STEPS):
            # env.render()
            observation = np.reshape(observation,(3,1))
            # print(observation.shape)
            action, next_state = agent.percieve(observation, next_state)
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            # print('step',step, 'action', action, 'observation', observation)

            if done:
                break

        # print("Episode {} = {}".format(episode, episode_reward))
        total_reward += episode_reward

    # print(agent.get_weights())

    # returns the average reward for number of episodes run
    total_reward /= EPISODES

    print(total_reward)
    return total_reward


def evaluateTester(individual):
    print(-np.sum(np.abs(individual)))
    return [-np.sum(np.abs(individual))]


def cross(winner, loser):
    """Apply a crossover operation on input sets. The first child is the
    intersection of the two sets, the second child is the difference of the
    two sets.

    Is ind1 the winner or ind2???
    """

    for i in range(NUM_PARAMS):
        if np.random.rand() < CROSS_PROB:
            loser[i] = winner[i]
    return loser


def mutate(individual):
    """Adds or subtracts 1% with a chance of 1/NUM_PARAMS"""
    # TODO: Increase speed of mutation
    for i in range(NUM_PARAMS):
        if np.random.rand() < (1/NUM_PARAMS):
            individual[i] += individual[i] * (np.random.rand()-0.5)*0.01
    return individual

toolbox.register("evaluate", evaluate)
toolbox.register("crossover", cross)
toolbox.register("mutate", mutate)
# TODO: create function to select 2 indiviuals within close proximity


def selDeme(individuals, deme_size):
    """Select *k* individuals at random from the input *individuals* with
    replacement. The list returned contains references to the input
    *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.

    This function uses the :func:`~random.choice` function from the
    python base :mod:`random` module.
    """
    one = np.random.randint(POPULATION_SIZE)
    _next = np.random.randint(1, deme_size + 1)
    if np.random.rand() < 0.5: _next = -_next
    two = (one + _next) % POPULATION_SIZE
    return individuals[one], individuals[two]

# toolbox.register("select", tools.selRandom, k=2)
toolbox.register("select", selDeme, deme_size=DEME_SIZE)



hof = tools.HallOfFame(1, np.array_equal)  # Store the best 3 individuals

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)

logbook = tools.Logbook()
logbook.header = ['gen', 'nevals'] + stats.fields


def main():

    pop = toolbox.population(n=POPULATION_SIZE)
    CXPB, MUTPB = 0.5, 0.2

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        # Set fitness property of individual to fit.
        ind.fitness.values = fit

    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=0, **record)

    # Start Tournament
    for g in range(1, NUM_GEN+1):
        print(g)
        # Select the next generation individuals
        # Selecting uses an object reference so no need to put individuals back
        # TODO: Make selection function that returns a tuple of individuals,
        # TODO: the first, a winner, the second the loser.
        chosen = toolbox.select(pop)
        # print(chosen[0].fitness.getValues)
        # print(chosen[1].fitness.values)
        # select winner and loser
        if chosen[0].fitness > chosen[1].fitness:
            winner = chosen[0]
            loser = chosen[1]
        elif chosen[1].fitness > chosen[0].fitness:
            winner = chosen[1]
            loser = chosen[0]
        else:
            continue
        del loser.fitness.values
        # print(winner.fitness.values)
        # print(winner.fitness.valid)

        # Apply crossover
        toolbox.crossover(winner, loser)

        # Apply mutation
        toolbox.mutate(loser)

        loser.fitness.values = toolbox.evaluate(loser)

        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=g, nevals=0, **record)

        # print(loser.fitness.values)
        #
        # # Apply crossover and mutation on the offspring
        # for child1, child2 in zip(offspring[::2], offspring[1::2]):
        #     if random.random() < CXPB:
        #         toolbox.mate(child1, child2)
        #         del child1.fitness.values
        #         del child2.fitness.values
        #
        # for mutant in offspring:
        #     if random.random() < MUTPB:
        #         toolbox.mutate(mutant)
        #         del mutant.fitness.values
        #
        # # Evaluate the individuals with an invalid fitness
        # invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # fitnesses = map(toolbox.evaluate, invalid_ind)
        # for ind, fit in zip(invalid_ind, fitnesses):
        #     ind.fitness.values = fit
        #
        # # The population is entirely replaced by the offspring
        # pop[:] = offspring

    return pop, logbook, hof

if __name__ == "__main__":
    # pop, logbook, hof = main()
    # pprint(logbook)
    #
    # # Best controller
    # print(hof.items[0])

    # Save best as CSV
    # For some reason saves under _practice package
    # np.savetxt("./bestController_fully_10nodes2000.csv", hof.items[0],
    #            delimiter=",")
    # agent.set_weights(hof.items[0])
    # np.savetxt("./bestController_bu.csv", hof.items[0], delimiter=",")
    # agent.set_weights(hof.items[0])

    agent.set_weights(np.loadtxt("./bestController_fully_10nodes2000.csv", delimiter=","))


    from evaluate import test
    print(test(agent=agent))

