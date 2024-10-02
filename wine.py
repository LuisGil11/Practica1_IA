import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn import linear_model


wine = load_wine()

wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
