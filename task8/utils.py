import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_labels(path):
    result = {}
    data = pd.read_csv(path)
    for i in range(len(data)):
        row = data.loc[i]
        name = row['filename']
        values = row[data.columns[1:]].values
        result[name] = int(values.astype('int')[0])
    return result


def vizualize(img, transpose = True, to_torch = False):
    if to_torch:
        img = torch.from_numpy(img)
    if transpose:
        img = img.permute(1, 2, 0)
    plt.imshow(img) #.int()
    plt.show()