import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import math

COLORS = ['red','blue','green','yellow','black','white','cyan','magenta','orange','purple','brown','pink','gray','olive','lime']

def show_figure():
    plt.show()
    plt.clf()

def pick_color():
    return COLORS.pop()

def distance(point, centroids):
    distances = np.zeros(len(centroids))
    for i in range(len(point)):
        for j in range(len(centroids)):
            distances[j] = (point[i] - centroids[j][i]) ** 2 + distances[j]

    distances = np.sqrt(distances)
    return distances


def k_means(k, points):
    centroids_with_colors = []
    centroids = np.random.standard_normal((k,2))
    show_figure()
    for centroid in centroids:
        color = pick_color()
        centroids_with_colors.append({
            "centroid": centroid,
            "color": color
        }) 
    
    for point in points:
        distances = distance(point, centroids)
        min_index = np.argmin(distances)
        closest_centroid = centroids_with_colors[min_index]
        plt.scatter(point[0], point[1], c=closest_centroid["color"])
        
np.random.seed(7)

x1 = np.random.standard_normal((100,2))*0.6+np.ones((100,2))
x2 = np.random.standard_normal((100,2))*0.5-np.ones((100,2))
x3 = np.random.standard_normal((100,2))*0.4-2*np.ones((100,2))+5
X = np.concatenate((x1,x2,x3),axis=0)

plt.plot(X[:,0],X[:,1],'k.')
show_figure()

k_means(3, X)
show_figure()





    