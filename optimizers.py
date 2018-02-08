from deap import creator
from deap import base
from deap import tools
import numpy as np
import random
import gym
from rnn import Basic_rnn, FullyConnectedRNN
from pprint import pprint
import environment

# ------------------------------------------------------------------------------
#                        SET UP: OpenAI Gym Environment
# ------------------------------------------------------------------------------
class MicrobialGA:

    def __init__(self, env, agent, EPISODES=3, STEPS=200, POPULATION_SIZE=16,
             CROSS_PROB=0.5, NUM_GEN=2000, DEME_SIZE=3 ):

        # self.ENV_NAME = 'Pendulum-v0'
        self.EPISODES = EPISODES  # Number of times to run envionrment when evaluating
        self.STEPS = STEPS  # Max number of steps to run run simulation
        self.env = env
        #
        # # Used to create controller
        # obs_dim = env.observation_space.shape[0]  # Input to controller (observ.)
        # action_dim = env.action_space.shape[0]  # Output from controller (action)
        # nodes = 10  # Unconnected nodes in network in the
        # dt = 0.05  # dt for the environment, found in environment source code
        # dt = 1 # Works the best, 0.05 causes it to vanish

        # ----------------------------------------------------------------------
        #                          SET UP: TensorFlow Basic_rnn
        # ----------------------------------------------------------------------
        self.agent = agent

        # ----------------------------------------------------------------------
        #                               SET UP GA PARAMETERS
        # ----------------------------------------------------------------------
        self.POPULATION_SIZE = POPULATION_SIZE
        self.CROSS_PROB = CROSS_PROB
        self.NUM_GEN = NUM_GEN   # Number of generations
        self.DEME_SIZE = DEME_SIZE  # from either side

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

        self.toolbox = base.Toolbox()

        # Create a function 'attr_item' to return the 'ID' of one item
        # num_nodes = num_inputs+1 = 20
        self.NUM_PARAMS = self.agent.num_params

        # create a function 'individual'. It takes an individual and instantiates
        #   it with a numpy array of size, NUM_PARAMS and type float32
        # Set the weights to a value n where, -0.1 < n < 0.1
        self.toolbox.register("attr_floats", lambda n: (np.random.rand(
            n).astype(
            np.float32)-0.5)*2, self.NUM_PARAMS)

        self.toolbox.register("individual", tools.initIterate,
                          creator.Individual, self.toolbox.attr_floats)

        # create a function 'population' that generates a list of individuals, x long.
        #   NB: As repeat init takes 3 parameters, and as the first 2 have been given.
        #       only one is needed.
        self.toolbox.register("population", tools.initRepeat, list,
                          self.toolbox.individual)

        # ----------------------------------------------------------------------
        # Register all functions from below and more
        # ----------------------------------------------------------------------

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("crossover", self.cross)
        self.toolbox.register("mutate", self.mutate)
        # toolbox.register("select", tools.selRandom, k=2)
        self.toolbox.register("select", self.selDeme, deme_size=DEME_SIZE)

        self.hof = tools.HallOfFame(1,
                               np.array_equal)  # Store the best 3 individuals

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)

        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals'] + self.stats.fields

    # ------------------------------------------------------------------------------
    #                         Evaluation function of GA
    # ------------------------------------------------------------------------------
    def evaluate(self, individual, MAX_REWARD=0):
        """Lends heavily from evaluate.py"""
        # Load weights into RNN
        self.agent.set_weights(individual)

        total_reward = 0
        for episode in range(self.EPISODES):
            # print("Starting Episode: {}".format(episode))

            # Starting observation
            observation = self.env.reset()

            episode_reward = 0
            next_state = np.random.rand(self.agent.state_size, 1).astype(
                dtype=np.float32)
            for step in range(self.STEPS):
                # env.render()
                observation = np.reshape(observation,(3,1))
                # print(observation.shape)
                action, next_state = self.agent.percieve(observation,
                                                         next_state)
                observation, reward, done, _ = self.env.step(action)
                episode_reward += reward
                # print('step',step, 'action', action, 'observation', observation)

                if done:
                    break

            # print("Episode {} = {}".format(episode, episode_reward))
            total_reward += episode_reward

        # print(self.agent.get_weights())

        # returns the average reward for number of episodes run
        total_reward /= self.EPISODES

        print(total_reward)
        return total_reward

    def evaluateTester(self, individual):
        print(-np.sum(np.abs(individual)))
        return [-np.sum(np.abs(individual))]

    def cross(self, winner, loser):
        """Apply a crossover operation on input sets. The first child is the
        intersection of the two sets, the second child is the difference of the
        two sets.

        Is ind1 the winner or ind2???
        """

        for i in range(self.NUM_PARAMS):
            if np.random.rand() < self.CROSS_PROB:
                loser[i] = winner[i]
        return loser

    def mutate(self, individual):
        """Adds or subtracts 1% with a chance of 1/NUM_PARAMS"""
        # TODO: Increase speed of mutation
        for i in range(self.NUM_PARAMS):
            if np.random.rand() < (1/self.NUM_PARAMS):
                individual[i] += individual[i] * (np.random.rand()-0.5)*0.01
        return individual

    # TODO: create function to select 2 indiviuals within close proximity

    def selDeme(self, individuals, deme_size):
        """Select *k* individuals at random from the input *individuals* with
        replacement. The list returned contains references to the input
        *individuals*.

        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :returns: A list of selected individuals.

        This function uses the :func:`~random.choice` function from the
        python base :mod:`random` module.
        """
        one = np.random.randint(self.POPULATION_SIZE)
        _next = np.random.randint(1, deme_size + 1)
        if np.random.rand() < 0.5: _next = -_next
        two = (one + _next) % self.POPULATION_SIZE
        return individuals[one], individuals[two]


    def run(self,verbosity=1):
        if(verbosity>0): print('Initializing population.')
        pop = self.toolbox.population(n=self.POPULATION_SIZE)
        CXPB, MUTPB = 0.5, 0.2

        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            # Set fitness property of individual to fit.
            ind.fitness.values = fit

        self.hof.update(pop)
        record = self.stats.compile(pop)
        self.logbook.record(gen=0, nevals=0, **record)

        # Start Tournament
        for g in range(1, self.NUM_GEN+1):
            if(verbosity>0): print('Running generation {}'.format(g))
            # Select the next generation individuals
            # Selecting uses an object reference so no need to put individuals back
            # TODO: Make selection function that returns a tuple of individuals,
            # TODO: the first, a winner, the second the loser.
            chosen = self.toolbox.select(pop)
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
            self.toolbox.crossover(winner, loser)

            # Apply mutation
            self.toolbox.mutate(loser)

            loser.fitness.values = self.toolbox.evaluate(loser)

            self.hof.update(pop)
            record = self.stats.compile(pop)
            self.logbook.record(gen=g, nevals=0, **record)

        if(verbosity>0): print('Completed optimization.')
        return pop

# Test the MicrobialGA algorithm works
if __name__ == "__main__":
    from rnn import FullyConnectedRNN
    import gym
    import tensorflow as tf

    n = 10  # number of nodes to test
    ENV_NAME = 'Pendulum-v0'
    env = gym.make(ENV_NAME)
    EPISODES = 3  # Number of times to run envionrment when evaluating
    STEPS = 200  # Max number of steps to run run simulation
    observation_space = env.observation_space.shape[
     0]  # Input to controller (observ.)
    action_space = env.action_space.shape[0]  # Output from controller (action)
    agent = FullyConnectedRNN(observation_space,action_space,n)
    opti = MicrobialGA(env, agent, EPISODES=1, NUM_GEN=100)
    opti.run()