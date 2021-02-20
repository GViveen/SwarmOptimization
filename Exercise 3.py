from ast import parse
from classes import Particle, Population, Clustering, fitness
import numpy as np
import sklearn.cluster as cluster
from sklearn import datasets
import matplotlib.pyplot as plt # for some reason says can't find the module on my pc, idk why.

# helper functions
def createArtificialProblem1():
    return np.array([[i[0], i[1]] for i in zip(np.random.uniform(-1, 1, 400), np.random.uniform(-1, 1, 400))])
    
def labelFunc1(x):
    x1 = x[0]
    x2 = x[1]
    return int((x1 >= 0.7) or ((x1<=0.3) and (x2 >= (-0.2 - x1))))

# problem data
X = createArtificialProblem1()
y = np.array([labelFunc1(x) for x in X])
clustering = Clustering(X, y)

# Iris data
iris = datasets.load_iris()
x_iris = iris.data
y_iris = iris.target
iris_clustering = Clustering(x_iris, y_iris)

#create particles
particles = []
pop_size = 50
particle_size = 2
rng = np.random.default_rng()
for i in range(pop_size):
    centroids = []
    for j in range(particle_size):
        centroids.append(np.array([ np.random.uniform(-1, 1, 1) for i in range(len(X[0]))]))
    particles.append(Particle(centroids, clustering.QuantizationError))
    
#create population
population = Population(particles, clustering.QuantizationError)

print("running...")
errors = []
n_iterations = 25
# test
for i in range(n_iterations):
    population.update_pop(r1=rng.random(1)[0], r2=rng.random(1)[0])
    errors.append(clustering.QuantizationError(population.global_best))
    
plt.figure(1)
plt.plot(errors)
plt.title("Quantization error after each iteration for the PSO clustering algorithm on the Artificial Dataset.")
plt.xlabel("Iteration")
plt.ylabel("Quantization Error")
plt.show()

pso_errors = errors

# Run k_means

kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

kmeans_cluster = Clustering(X, y)
kmeans_error = kmeans_cluster.QuantizationError(centroids)
print("k_means final error on artificial: {}".format(kmeans_error))
print("PSO final error on artificial after {}: {}".format(n_iterations, pso_errors[-1]))

# Do the same for iris
particles = []
pop_size = 50
particle_size = 4
for i in range(pop_size):
    centroids = []
    for j in range(particle_size):
        centroids.append(np.array([rng.uniform(0, np.max(x_iris)) for k in range(len(x_iris[0]))]))
    particles.append(Particle(centroids, iris_clustering.QuantizationError))

population = Population(particles, iris_clustering.QuantizationError)

errors = []
n_iterations = 50

for i in range(n_iterations):
    population.update_pop(r1=rng.random(1)[0], r2=rng.random(1)[0])
    errors.append(iris_clustering.QuantizationError(population.global_best))
    
plt.figure(2)
plt.plot(errors)
plt.title("Quantization error after each iteration for the PSO clustering algorithm on the Iris Dataset.")
plt.xlabel("Iteration")
plt.ylabel("Quantization Error")
plt.show()

pso_errors = errors

# Run k_means

kmeans = cluster.KMeans(n_clusters=4, random_state=0).fit(x_iris)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

kmeans_cluster = Clustering(x_iris, y_iris)
kmeans_error = kmeans_cluster.QuantizationError(centroids)
print("k_means final error on iris: {}".format(kmeans_error))
print("PSO final error on iris after {}: {}".format(n_iterations, pso_errors[-1]))