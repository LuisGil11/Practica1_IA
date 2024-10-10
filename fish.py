import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def normalize(data: list[float]) -> list[float]:
    mean = sum(data) / len(data)
    std_dev = (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
    return [(x - mean) / std_dev for x in data], mean, std_dev

def squared_error(y: list[float], y_pred: list[float]) -> float:
    return sum((yi - y_predi) ** 2 for yi, y_predi in zip(y, y_pred)) / (2 * len(y))

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

    # Normalizamos x e y
    x, x_mean, x_std = normalize(x)
    y, y_mean, y_std = normalize(y)

    for i in range(iterations):

        y_pred = [m * xi + b for xi in x]

        # Paso 1. Calculamos las derivadas parciales para obtener el gradiente
        dm = (-1 / (2 * n)) * sum((yi - y_predi) * xi for yi, y_predi, xi in zip(y, y_pred, x))
        db = (-1 / (2 * n)) * sum(yi - y_predi for yi, y_predi in zip(y, y_pred))

        # Paso 2. Actualizar los parámetros
        m -= learning_rate * dm
        b -= learning_rate * db

        # 3. Calcular el error cuadrático medio
        mse = squared_error(y, y_pred)

        # 4. Condición de parada basada en el error cuadrático medio
        if mse < tolerance:
            break
    
    # Desnormalizar los parámetros
    m_desnormalizado = m * (y_std / x_std)
    b_desnormalizado = b * y_std + y_mean - m_desnormalizado * x_mean

    print(f'Error cuadrático medio: {mse}, despues de f{i + 1} iteraciones')

    return m_desnormalizado, b_desnormalizado


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

m_mse, b_mse = lineal_regression_by_min_square_errors(X, Y)

print(f'm = {m_mse}')
print(f'b = {b_mse}')

# Generar puntos para la recta
x_range = np.linspace(min(X), max(X), 100)
y_range = m_mse * x_range + b_mse

# Graficar los puntos originales y la recta de regresión
plt.scatter(X, Y, color='blue', label='Datos originales')
plt.plot(x_range, y_range, color='red', label=f'Recta de regresión: y = {m_mse:.2f}x + {b_mse:.2f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regresión Lineal por Mínimos Cuadrados')
plt.legend()
plt.show()

m_gd, b_gd = gradient_descend(X, Y, 0.01, 1000, 0.001)

print(f'm ={m_gd}')
print(f'b ={b_gd}')

# Graficar los puntos originales y la recta de regresión
plt.scatter(X, Y, color='blue', label='Datos originales')
plt.plot(x_range, y_range, color='red', label=f'Recta de regresión por descenso de gradiente: y = {m_gd:.2f}x + {b_gd:.2f}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regresión Lineal por Descenso de Gradiente')
plt.legend()
plt.show()

# m, b = stochastic_gradient_descent(X, Y)


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