import pandas as pd
from NeuralNetwork.Training_structre import training, splitting_data
from NeuralNetwork.getData import data_preprocessing
from NeuralNetwork.Network_single_class1 import NNetwork
from Data.Transformers import StandardizationType
import multiprocessing
#
import numpy as np

# # stworzenie modelu i szkolenie go na danych data
data = pd.read_csv(r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\new_data.csv",delimiter=";")

data = data_preprocessing([1, 6, 7, 8, 12,4],[2, 9, 10, 11],[],StandardizationType.Z_SCORE,True)#  False,data
import random
random.seed(32)
network = NNetwork(epoki=80, alpha=0.001, optimizer="RMSprop",
                   gradients="mini-batch")  # optimizer="momentum",gradients="batch"
network.add_layer(31, 12, "relu",seed_for_all=32)
network.add_layer(12, 12, "relu",seed_for_all=32)
network.add_layer(12, 1, "sigmoid",seed_for_all=32)
x_train, y_train, x_valid, y_valid, x_test, y_test = splitting_data(data)
network =training(data,x_test,y_test ,network =network,batch_size=32,range_i =1)

#data = pd.read_csv(r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\training_data.csv",delimiter=";")

# nn = NNetwork.create_instance(alfa=0.000001)

#

# training(data,x_test,y_test ,network =nn,batch_size=32,range_i =1)
#
#
# X = data.iloc[:,:-1].values
# Y = data.iloc[:,-1].values
#
#
#
#
# nn.after(show=True)
# predictions= []
# print(nn.perceptron(X,Y))
# for x_point,y_point in zip(X,Y):
#     out = nn.pred(x_point)[0]
#     y_pred = 1 if out>=0.51 else 0
#     #print(out, y_pred,y_point)
#     predictions.append(y_pred)
# print(nn.confusion_matrix(predictions,Y.tolist()))

#
#data = data_preprocessing([1, 6, 7, 8, 12,4],[2, 9, 10, 11],[],StandardizationType.NORMALIZATION,True)#  False,data
