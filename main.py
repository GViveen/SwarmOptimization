from ast import parse
from classes import Particle, Population, Clustering
import numpy as np


# helper functions
def createArtificialProblem1():
    """Dataset for artifical problem 1 from paper

    Returns:
        [array]: 2d array of float from uniform distribution between -1 and 1
    """
    return np.array([[i[0], i[1]] for i in zip(np.random.uniform(-1, 1, 400), np.random.uniform(-1, 1, 400))])


def labelFunc1(x):
    """ Labelizes data for artifical problem 1 from paper

    Args:
        x (array): 1d array of values to label

    Returns:
        [int]: label for data x.
    """
    x1 = x[0]
    x2 = x[1]
    return int((x1 >= 0.7) or ((x1 <= 0.3) and (x2 >= (-0.2 - x1))))


def UCI_dataset():
    """ Reads Iris dataset presented at: https://archive.ics.uci.edu/ml/datasets/iris.

    Returns:
        x (array): array of values
        y (array): array of labels
    """
    file = open("iris.data", "r")
    data = file.read()

    x = []
    y = []
    for d in data.split():
        x.append([float(n) for n in d.split(',')[:-1]])
        y.append(d.split(',')[-1])
    return np.array(x), np.array(y)


def run_clustering(X, y, iterations):
    clustering = Clustering(X, y)

    # create particles
    particles = []
    pop_size = 10
    particle_size = 2
    for i in range(pop_size):
        centroids = []
        for j in range(particle_size):
            centroids.append(np.array([np.random.uniform(-1, 1, 1) for i in range(len(X[0]))]))
        particles.append(Particle(centroids, clustering.QuantizationError))

    # create population
    population = Population(particles, clustering.QuantizationError)

    errors = []
    # test
    for i in range(iterations):
        print("-----------Run {}-----------".format(i))
        population.update_pop()
        error = clustering.QuantizationError(population.global_best)
        print(error)
        errors.append(error)
    return errors


if __name__ == "__main__":
    # problem data

    iterations = 30
    X = createArtificialProblem1()
    y = np.array([labelFunc1(x) for x in X])

    error_A1 = run_clustering(X, y, 30)

    X, y = UCI_dataset()
    error_Iris = run_clustering(X, y, 30)

    print("Mean error A1:", sum(error_A1) / len(error_A1))
    print("Mean error Iris:", sum(error_Iris) / len(error_Iris))

