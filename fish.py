import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def squared_error(y: list[float], y_pred: list[float]) -> float:
    n = len(y)
    return sum((y - y_pred) ** 2) / (2 * n)

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

        y_pred = m * x + b

        # Paso 1. Calculamos las derivadas parciales para obtener el gradiente
        dm = (-1 / 2* n) * sum((y - y_pred) * x)
        db = (-1 / 2* n) * sum(y - y_pred)

        print(f'dm = {dm}, db = {db}')

        # Paso 2. Actualizar los parámetros
        m -= learning_rate * dm
        b -= learning_rate * db

        # 3. Calcular el error cuadrático medio
        mse = squared_error(y, y_pred)

        # 4. Condición de parada basada en el error cuadrático medio
        if mse < tolerance:
            break

    return m, b


def lineal_regression_by_min_square_errors(x: list[float], y: list[float]):
    n = len(x)

    # Calculamos los valores de las constantes
    k1 = sum(x * y)
    k2 = sum(x ** 2)
    k3 = sum(x)
    k4 = sum(y)
    
    # Sacamos m y b
    m = - ((k1 * n) - (k3 * k4)) / (k3 ** 2 - (k2 * n))
    b = - ((k2 * k4) - (k1 * k3)) / (k3 ** 2 - k2 * n)

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

# m, b = gradient_descend(X, Y, 0.001, 1000, 0.001)

# print(f'm ={m}')
# print(f'b ={b}')

# m, b = stochastic_gradient_descent(X, Y)

# print(f'm ={m}')
# print(f'b ={b}')

# Dividimos la data en data de entrenamiento y data de prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Obtenemos la pendiente y el punto de corte del modelo a partir de los datos de entrenamiento
m_min_sq, b_min_sq = lineal_regression_by_min_square_errors(X_train, Y_train)

Y_pred_min_sq = m_min_sq * X_test + b_min_sq

# Calculamos el error cuadrático medio para el modelo de regresión de mínimos cuadrados
mse_min_sq = squared_error(Y_test, Y_pred_min_sq)
print(f'Error cuadrático medio para regresión de mínimos cuadrados: {mse_min_sq}')


# m_grad_desc, b_grad_desc = gradient_descend(X_train, Y_train, 0.001, 1000, 0.001)
# m_est, b_est = stochastic_gradient_descent(X_train, Y_train)

linear_regression = LinearRegression()
linear_regression.fit(X_train.values.reshape(-1, 1), Y_train)

Y_pred = linear_regression.predict(X_test.values.reshape(-1, 1))

# Calculamos el error cuadrático medio para el modelo de regresión lineal de sklearn
mse = squared_error(Y_test, Y_pred)
print(f'Error cuadrático medio para regresión lineal de sklearn: {mse}')