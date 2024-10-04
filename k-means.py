import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.datasets import load_wine
from sklearn.cluster import KMeans

def elbow_method(data, max_clusters=10):
    """
    
    El método del codo consiste en encontrar el número de clústeres para K-means. 
    
    Este método se basa en identificar el número de clústeres para el que se observa
    un cambio significativo en la tasa de disminución de la varianza intracluster.

    Para ello se ejecuta k-means con diferentes números de clústeres y se calcula la 
    suma de las distancias al cuadrado para cada punto con respecto a su centroide.
    Usando los resultados para crear una gráfica con los valores de k en el eje x y 
    la suma de las distancias al cuadrado en el eje y. En esta gráfica se busca el punto
    donde se produce un cambio brusco en la disminución de la suma de las distancias al
    cuadrado.

    """
    sum_of_squared_distances = []
    for k in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        sum_of_squared_distances.append(kmeans.inertia_)
    
    plt.plot(range(1, max_clusters+1), sum_of_squared_distances, 'bx-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Suma de las Distancias al Cuadrado')
    plt.title('Curva del Codo')
    plt.show()

def set_graphic_info():
    plt.xlabel('Alcohol')
    plt.ylabel('Magnesium')
    plt.title('Scatter plot of Alcohol vs Magnesium')

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
    return COLORS.pop(0)

def min_distances(point: Point, centroids: list[Centroid]):
    distances = []
    for i in range(len(centroids)):
        distances.append(point.distance_to(centroids[i].position))
    return distances

def k_means(k, points: list[Point]):
    centroids: list[Centroid] = []
    x_positions = 10 + 8 * np.random.rand(k)  # x en el rango [10, 18]
    y_positions = 60 + 120 * np.random.rand(k)  # y en el rango [60, 180]
    centroids_position = np.column_stack((x_positions, y_positions))

    for position in centroids_position:
        color = pick_color()
        centroid = Centroid(position, color)
        centroids.append(centroid)
        plt.scatter(centroid.position[0], centroid.position[1], c=centroid.color, marker='x')

    for _ in range(12):
        plt.clf()
        for point in points:
            distances = min_distances(point, centroids)
            
            min_index = np.argmin(distances)
            closest_centroid = centroids[min_index]
            point.centroid = closest_centroid
            centroids[min_index].addPoint(point)
            plt.scatter(point.position[0], point.position[1], c=point.centroid.color)
        
        for centroid in centroids:
            centroid.recenter()
            plt.scatter(centroid.position[0], centroid.position[1], c=centroid.color, marker='x')
        
        set_graphic_info()
        plt.show()

wine = load_wine()

data = pd.DataFrame(data=np.c_[wine['data'], wine['target']], columns=wine['feature_names']+['target'])

elbow_method(data, 10)

df = data[['alcohol', 'magnesium']]

plt.scatter(df['alcohol'], df['magnesium'], c='k', marker='.')
set_graphic_info()
plt.show()

wine_data = []

[wine_data.append(Point([float(coordinates['alcohol']), float(coordinates['magnesium'])])) for coordinates in df.iloc]

k_means(3, wine_data)





    