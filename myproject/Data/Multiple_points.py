from random import random, choice

import pandas as pd

def multiply(data, times, rate):
    """
    :param  data is x and y DataFrame with data to multiply
    :param times  defines how nany rows like given are created
    :param rate random + - percentage  threshold of a current point
    :return [data] + created_data X [times] with [rate]% of threshold in pd.DataFrame"""
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
     :param rate  is default = 0.1 is the rate of change value 0.1 is equivalent to 10% max change
    """
    return wspolrzedna + (random() * wspolrzedna * rate * choice([-1, 1]))




#
# from Multiple_points import multiply
# data = pd.DataFrame([[1,2,3,0]],columns=["a","b","c","y"])
# print("data before: ",data.values)
# print("data after multiply 4 times without error ")
#
# xmult = multiply(data,4,0)
# print(xmult.values)