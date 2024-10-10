import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

def stochastic_gradient_descent(x: list[float], y: list[float], learning_rate: float = 0.001, iterations: int = 1000):
    # Inicializamos los parámetros
    m = 0
    b = 0
    n = len(x)

    for _ in range(iterations):
        i = random.randint(0, n - 1)
        xi = x[i]
        yi = y[i]
        y_pred = xi * m + b
        error = yi - y_pred

        # Calcular los gradientes, no dividimos por n ya que estamos tomando un solo punto
        dm = - 2 * xi * error
        db = - 2 * error

        m -= learning_rate * dm
        b -= learning_rate * db

        # Calcular el error cuadrático medio (MSE)
        y_preds = [m * xj + b for xj in x]
        square_error = sum((yj - y_predj) ** 2 for yj, y_predj in zip(y, y_preds)) / n

        if square_error < 0.001:
            break

    return m, b

def gradient_descend(x: list[float], y: list[float], learning_rate: float = 0.001, iterations: int = 1000, tolerance: float = 0.001):
    # Inicializamos los parámetros
    m = 0
    b = 0
    n = len(x)

    for _ in range(iterations):

        # Paso 1. Calcular los gradientes
        dm = - (1/2 * n) * sum((y - m * x - b) * x)
        db = - (1/2 * n) * sum(y - m * x - b)

        # Paso 2. Actualizar los parámetros
        m -= learning_rate * dm
        b -= learning_rate * db

        # 3. Calcular el error cuadrático medio
        square_error = sum((y - m * x - b) ** 2) / (2 * n)

        # 4. Condición de parada basada en el error cuadrático medio
        if square_error < tolerance:
            break

    return m, b


def lineal_regression_by_min_square_errors(x: list[float], y: list[float]):
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

X = data['Weight']
Y = data['Height']

# Paso 4: Graficar los datos
plt.scatter(X, Y, color='blue', label='Datos originales')
plt.xlabel('Weight')
plt.ylabel('Height')
plt.title('Scatter Plot de Weight vs Height para la especie Bream')

plt.show()

m, b = lineal_regression_by_min_square_errors(X, Y)

print(f'm = {m}')
print(f'b = {b}')

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


m, b = gradient_descend(X, Y, 1.22e-5, 1000, 0.001)

print(f'm ={m}')
print(f'b ={b}')

# m, b = stochastic_gradient_descent(X, Y)

# print(f'm ={m}')
# print(f'b ={b}')