# Testy wszystkich funkcjonalności złożonych z danych
import random

import flask.cli

from Data.Transformers import Transformations
from Data.Multiple_points import multiply
from Data.Transformers import Transformations,StandardizationType
from Data.SPLIT_test_valid_train import SplitData
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
x={"kasza":["a","b","c","d","e","e"],
   "masza":["kura","dziura","kinga","dziura","małpa","małpa"]
   }
y = [1,1,0,5,5,0]
new_d = pd.Series(["średnia","wysoka","niska","wysoka","niska","niska"],name="temperature")
d = pd.DataFrame(x)
d["y"] = y
d =pd.concat((d,new_d),axis=1)
one_h = OneHotEncoder()
print()

one_h.code_keys(d.iloc[:, :-2])
one_h.label_encoder_keys(d.iloc[:, [-1, -2]], [["niska", "średnia", "wysoka"], [5, 1, 0]])
one_h.code_y_for_network(d)
new =pd.DataFrame([["a","kinga",5,"wysoka"]],columns=d.columns)

output = one_h.code_y_for_network(new)



#print(one_h.data_code)
one_h.save_data()
from flask import Flask
#print(flask.cli.version_option())

### data
to_data_frame = pd.read_csv(r"C:\Program Files\Pulpit\Data_science\Zbiory\iris_orig.csv",delimiter=",")
x = to_data_frame.iloc[:,:-1]
from Data.Transformers import Transformations,StandardizationType
norm = Transformations(x,StandardizationType.Z_SCORE)
x =norm.standarization_of_data(x).values



y = to_data_frame.iloc[:,-1]
from Data.One_hot_Encoder import OneHotEncoder
onehot = OneHotEncoder()
onehot.label_encoder_keys(y,[['Iris-versicolor','Iris-setosa','Iris-virginica']])
y = onehot.code_y_for_network(y).values