import numpy as np
import pandas as pd


def load_labels(path):
    result = {}
    data = pd.read_csv(path)
    for i in range(len(data)):
        row = data.loc[i]
        name = row['filename']
        values = row[data.columns[1:]].values
        result[name] = int(values.astype('int')[0])
    return result