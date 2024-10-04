import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import random

class Point:
    def __init__(self, coordinates) -> None:
        self.position = coordinates
        self.centroid = None

    def distance_to(self, coordinates):
        distance = 0
        for i in range(len(self.position)):
            distance = (self.position[i] - coordinates[i]) ** 2 + distance
        return math.sqrt(distance)
        
class Centroid:
    def __init__(self, position, color) -> None:
        self.position = position
        self.color = color
        self.points: list[Point] = []

    def recenter(self):
        new_position = [0, 0]
        for point in self.points:
            new_position[0] += point.position[0]
            new_position[1] += point.position[1]
        new_position[0] = new_position[0] / len(self.points)
        new_position[1] = new_position[1] / len(self.points)
        self.position = new_position
    
    def addPoint(self, point: Point):
        self.points.append(point)



COLORS = ['red','blue','green','yellow','black','white','cyan','magenta','orange','purple','brown','pink','gray','olive','lime']


def pick_color():
    return COLORS.pop()

def min_distances(point: Point, centroids: list[Centroid]):
    distances = []
    for i in range(len(centroids)):
        distances[i] = 0
    for i in range(len(centroids)):
        distances[i] = point.distance_to(centroids[i].position)
    return distances

def k_means(k, points: list[Point]):
    centroids: list[Centroid] = []
    centroids_position = random.random((k,2))
    for position in centroids_position:
        color = pick_color()
        centroid = Centroid(position, color)
        centroids.append(centroid)
        plt.scatter(centroid.position[0], centroid.position[1], c=centroid.color, marker='x')

    for _ in range(2):
        for point in points:
            distances = min_distances(point, centroids)
            
            min_index = np.argmin(distances)
            closest_centroid = centroids[min_index]
            point.centroid = closest_centroid
            centroids[min_index].addPoint(point)
            plt.scatter(point.position[0], point.position[1], c=point.centroid.color)
        plt.show()
        plt.clf()
        for centroid in centroids:
            centroid.recenter()
            plt.scatter(centroid.position[0], centroid.position[1], c=centroid.color, marker='x')
    plt.show()
    plt.clf()
        
np.random.seed(7)

x1 = np.random.standard_normal((100,2))*0.6+np.ones((100,2))
x2 = np.random.standard_normal((100,2))*0.5-np.ones((100,2))
x3 = np.random.standard_normal((100,2))*0.4-2*np.ones((100,2))+5
X = np.concatenate((x1,x2,x3),axis=0)

points = []

for coordinates in X:
    points.append(Point(coordinates))


plt.plot(X[:,0],X[:,1],'k.')
plt.show()

k_means(3, points)





    