# Testy wszystkich funkcjonalności złożonych z danych

from Data.Transformers import Transformations
from Multiple_points import multiply
from Data.Transformers import Transformations,StandardizationType
import numpy as np
import pandas as pd
from Data.One_hot_Encoder import OneHotEncoder

#x = [[2.1,3.8,1.6],[4.5,3.1,3.2],[1.3,2.8,1.2],[2.1,1.0,0.5],[0.2,1.6,0.3],[0.4,2.1,2.61],[2.1,1.7,3.3]]
#y = [1,0,1,0,1,2,2]
x =[[0.2,1],[1.9,0.3]]
y = [0,1]
#create pandas data frame with x,y
data_frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(len(x[0])) ])
data_frame["y"] = y

# multipy data points
# poiwieleone 8 razy i threshold jest na poziomie +|- 0.01
data =multiply(data_frame,8,0.1)
y = data.loc[:,'y']
x = data.iloc[:,:-1]
#print(x)
#normalizacja danych metodą min, max
norma = Transformations(x,std_type=StandardizationType.NORMALIZATION)
x = norma.standarization_of_data(x)
print(x)

#normalizacja danych metodą mean_score
norma = Transformations(x,std_type=StandardizationType.NORMALIZATION)
x = norma.standarization_of_data(x)

#normalizacja danych metodą z_score
norma = Transformations(x,std_type=StandardizationType.NORMALIZATION)
x = norma.standarization_of_data(x)


#normalizacja danych metodą logarytmu
norma = Transformations(x,std_type=StandardizationType.NORMALIZATION)
x = norma.standarization_of_data(x)


# złączenie danych x i y
data_frame = pd.DataFrame(x)
data_frame["y"]=y
print(data_frame)
#print(data_frame)


# metoda dzieląca dane
from SPLIT_test_valid_train import *
#definiowanie  podziału  treningowe|validacyjne|testowe
sd = SplitData.set(0.5,0.3,0.2)
x_train,y_train,x_valid,y_valid,x_test,y_test = SplitData.split_data(data_frame)

print(x_train,y_train)
print()
print(x_valid,y_valid)
y =np.array(["banan",8,7,7,"a"])

# tworzenie batchy
x,y = SplitData.batch_split_data(data_frame, 4)
for x_,y_ in zip(x,y):
    print(x_,y_)


## testy one_hot_enocdoera kodowanie i tworzenie zbioru danych do treningu
x=["a","b","c","d","e","e"]
y = [1,1,0,0,0,0]
d = pd.DataFrame(x)
d["y"]=y
one_h = OneHotEncoder(d)
one_h.code_keys(x)
print(one_h.decoded_set)
print(one_h.label_code)
print(one_h.code_y_for_network(x))
#print(one_h.decode_keys([0.5,0.2,0.1,0.34,0.3]))
print(one_h.number_of_coded_keys)

