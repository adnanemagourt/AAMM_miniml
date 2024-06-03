import pandas as pd
import numpy as np

def load_adnane():
    data = pd.read_csv("adnane.csv")
    return { "data": data.iloc[:, :9].values, "target": data.iloc[:, 9].values }

def load_mahmoud():
    data = pd.read_csv("mahmoud.csv")
    return { "data": data.iloc[:, :9].values, "target": data.iloc[:, 9].values }


def generate_random_dataset(name):
    data = np.random.rand(1000, 9)
    target = np.random.randint(0, 2, 1000)
    return { "data": data, "target": target }

