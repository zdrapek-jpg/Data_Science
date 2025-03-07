from Multiple_points import multiply
from One_hot_Encoder import OneHotEncoder
from Transformers import *
import numpy as np

#x = [[2.1,3.8,1.6],[4.5,3.1,3.2],[1.3,2.8,1.2],[2.1,1.0,0.5],[0.2,1.6,0.3],[0.4,2.1,2.61],[2.1,1.7,3.3]]
#y = [1,0,1,0,1,2,2]
x =[[5,5],[4.9,1]]
y = [0,1]
#create pandas data frame with x,y
data_frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(len(x[0])) ])
data_frame["y"] = y

# multipy data points
data =multiply(data_frame,8,0.01)
y = data.loc[:,'y']
x = data.iloc[:,:-1]
#print(data)

#normalizacja danych
norma = Transformations(x)
x = norma.stadaryzacja_min_max_normalization(x)

# standaryzacja danych
#d =norma.standaryzacja_mean_score(x,y)
#d =norma.standaryzacja_danych(x.values,y)
#print(d)
#print(norma.srednie)
#print(norma.odchylenia_w_kolumnach)



data_frame = pd.DataFrame(x)
data_frame["y"]=y
#print(data_frame)
#print(data_frame)
#
from SPLIT_train_valid_test import *

sd = SplitData.set(0.5,0.3,0.2)
x_train,y_train,x_valid,y_valid,x_test,y_test = SplitData.split_data(data_frame)

print(x_train,y_train)
print()
print(x_valid,y_valid)
y =np.array(["banan",8,7,7,"a"])

one_hot = OneHotEncoder(y)

print(one_hot.code_keys(y))
print(one_hot.code_y_for_network(list(y)))

# rzykładowa odpowiedź modelu
y_pred = np.array([0.1,0.2,0.2,0.3,0.7])

print(one_hot.decode_keys(y_pred))

print(data_frame)

x,y = SplitData.batch_split_data(data_frame, 4)




