import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

class Centroid:
    def __init__(self, position, color) -> None:
        self.position = position
        self.color = color

class Point:
    def __init__(self, coordinates) -> None:
        self.position = coordinates
        self.centroid = None

    def distance_to(self, coordinates):
        distance = 0
        for i in range(len(self.position)):
            distance = (self.position[i] - coordinates[i]) ** 2 + distance
        return np.sqrt(distance)
        
COLORS = ['red','blue','green','yellow','black','white','cyan','magenta','orange','purple','brown','pink','gray','olive','lime']

def show_figure():
    plt.show()
    plt.clf()

def pick_color():
    return COLORS.pop()

def min_distance(point: Point, centroids: list[Centroid]):
    distances = np.zeros(len(centroids))
    for i in range(len(centroids)):
        distances[i] = point.distance_to(centroids[i].position)
    return distances

def k_means(k, points):
    centroids = []
    centroids_position = np.random.standard_normal((k,2))
    for position in centroids_position:
        color = pick_color()
        centroid = Centroid(position, color)
        centroids.append(centroid)
        plt.scatter(centroid.position[0], centroid.position[1], c=centroid.color, marker='x')


    for point in points:
        distances = min_distance(point, centroids)
        min_index = np.argmin(distances)
        closest_centroid = centroids[min_index]
        point.centroid = closest_centroid
        plt.scatter(point.position[0], point.position[1], c=point.centroid.color)
        
np.random.seed(7)

x1 = np.random.standard_normal((100,2))*0.6+np.ones((100,2))
x2 = np.random.standard_normal((100,2))*0.5-np.ones((100,2))
x3 = np.random.standard_normal((100,2))*0.4-2*np.ones((100,2))+5
X = np.concatenate((x1,x2,x3),axis=0)

points = []

for coordinates in X:
    points.append(Point(coordinates))


plt.plot(X[:,0],X[:,1],'k.')
show_figure()

k_means(3, points)
show_figure()





    