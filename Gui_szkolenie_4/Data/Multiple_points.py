from random import random, choice

import pandas as pd


def multiply(datax, times, rate):
    y1 = len(datax)
    for_pdframe_data = []
    for _ in range(times):
        for _, row in datax.iterrows():
            data_new = []
            for wspolrzedna in row[:-1]:
                data_new.append(randomize(wspolrzedna, rate))
            data_new.append(row[-1])
            for_pdframe_data.append(data_new)

    for_pdframe_data = pd.DataFrame(for_pdframe_data, columns=datax.columns)
    for_pdframe_data = pd.concat([datax, for_pdframe_data], ignore_index=True)
    return for_pdframe_data


def randomize(wspolrzedna, rate=0.1):
    return wspolrzedna + (random() * wspolrzedna * rate * choice([-1, 1]))
