import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def regression_by_min_square_errors(x: list[float], y: list[float]):
    k1 = 0
    k2 = 0
    k3 = 0
    k4 = 0

    # Calculamos los valores de las constantes
    for i in range(len(x)):
        k1 += x[i] * y[i]
        k2 += x[i] ** 2
        k3 += x[i]
        k4 += y[i]
    
    # Sacamos m y b
    m = - ((k1 * len(x)) - (k3 * k4)) / (k3 ** 2 - (k2 * len(x)))
    b = - ((k2 * k4) - (k1 * k3)) / (k3 ** 2 - k2 * len(x))

    return m, b

data = pd.read_csv("./Fish.csv")

# data.info()

data = data[data.Species == "Bream"]

X = data['Weight'].tolist()
Y = data['Height'].tolist()

# Paso 4: Graficar los datos
plt.scatter(X, Y, color='blue', label='Datos originales')
plt.xlabel('Weight')
plt.ylabel('Height')
plt.title('Scatter Plot de Weight vs Height para la especie Bream')

plt.show()

m, b = regression_by_min_square_errors(X, Y)

print(f'm ={m}')
print(f'b ={b}')

# Generar puntos para la recta
x_range = np.linspace(min(X), max(X), 100)
y_range = m * x_range + b

# Graficar los puntos originales y la recta de regresión
plt.scatter(X, Y, color='blue', label='Datos originales')
plt.plot(x_range, y_range, color='red', label=f'Recta de regresión: y = {m:.2f}x + {b:.2f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regresión Lineal por Mínimos Cuadrados')
plt.legend()
plt.show()