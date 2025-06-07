import numpy as np
import pandas as pd
def fill(pandas_frame):
    cols_integer = list(pandas_frame.select_dtypes(include=["number", "int", "float"]).columns)
    cols_strings = list(pandas_frame.select_dtypes(include=["string","object","category"]).columns)

    print(cols_integer)
    print(cols_strings)

    if cols_integer:
        pandas_frame.loc[1,cols_integer] = -21
        pandas_frame.loc[2,cols_integer[0]] = np.nan

    if cols_strings:
        pandas_frame.loc[[2, 3],cols_strings[-1]] = ""
        pandas_frame.loc[0,cols_strings[0:-1]] = None
    return pandas_frame


def count(data_frame):
    """
    :param data_frame:  fframe for check if data has only one value so we can skip it, if one variable has over 97% of rown count we can skip it becous there is too little data split in classes
    :return:  frame with deleten columns if it has any prblems
    """
    columns = data_frame.columns

    ### drop columns where missing values are over 85% of data
    data_frame = data_frame.dropna(thresh=0.85,axis=1)
    for col in columns:
        counted = {}
        in_colum_values = data_frame[col].unique()
        # jeśli kolumna ma dużo zmiennych to jej nawet nie analizujemy
        if len(in_colum_values)>20:
            continue
        ### iteracja po unikalnych obiektach kolumny i ich zliczanie
        for value in in_colum_values:

            ilosciowo =data_frame[col].value_counts()[value]
            counted[value]=ilosciowo
        if counted.__len__()==1:
            data_frame.drop(col,axis=1,inplace=True)
            print("drop: ", counted," kolumna: ", col)
            return data_frame
        if counted.__len__()<=3:
            # if any value has for example over 96% of record delete row becouse his
            if any(value>0.97*data_frame.shape[0] for value in counted.values()):
                data_frame.drop(col,axis=1,inplace=True)
                print("drop: ",counted," kolumna:",col)


    return data_frame



