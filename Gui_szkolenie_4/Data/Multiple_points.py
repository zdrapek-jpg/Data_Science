from random import random, choice

import pandas as pd

def multiply(data, times, rate):
    """
    :param  [data] is x and y DataFrame with data to multiply  [times] and threshold at [rate]%
    :return [data] + created_data X [times] with [rate]% of threshold
    """
    y1 = len(data)
    for_pdframe_data = []
    for _ in range(times):
        for _, row in data.iterrows():
            data_new = []
            for wspolrzedna in row[:-1]:
                data_new.append(randomize(wspolrzedna, rate))
            data_new.append(row[-1])
            for_pdframe_data.append(data_new)

    for_pdframe_data = pd.DataFrame(for_pdframe_data, columns=data.columns)
    for_pdframe_data = pd.concat([data, for_pdframe_data], ignore_index=True)
    return for_pdframe_data

def randomize(wspolrzedna, rate=0.1):
    """
     new point = point +/- threshold * point
    """
    return wspolrzedna + (random() * wspolrzedna * rate * choice([-1, 1]))
