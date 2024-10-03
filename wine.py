import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt


wine = load_wine()

# print(dir(wine))

data = pd.DataFrame(data=np.c_[wine['data'], wine['target']], columns=wine['feature_names']+['target'])

# print("data:")
# print(wine['data'])

# print("target:")
# print(wine['target'])

# print("feature_names:")
# print(wine['feature_names'])

# print(data.head())

# print(data.describe())

df = data[['alcohol', 'magnesium', 'color_intensity', 'target']]

# print(df.corr())

sns.pairplot(df, hue='target')

plt.show()