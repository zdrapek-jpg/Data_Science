import pandas as pd

from NeuralNetwork.Training_structre import training, splitting_data
from NeuralNetwork.getData import data_preprocessing
from NeuralNetwork.Network_single_class1 import NNetwork
from Data.Transformers import StandardizationType
#
#
import numpy as np
#
#
# # stworzenie modelu i szkolenie go na danych data
# data1 = pd.read_csv(r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\new_data.csv",delimiter=";")
# data1["y"] = np.where(data1["y"]=="no",0,1)
# data = data_preprocessing([1, 6, 7, 8, 12,4],[2, 9, 10, 11],[],StandardizationType.NORMALIZATION,False,data1)#  False,data
#
# network = NNetwork(epoki=99, alpha=0.003, optimizer="adam",
#                    gradients="mini-batch")  # optimizer="momentum",gradients="batch"
# network.add_layer(31, 12, "relu")
# network.add_layer(12, 12, "relu")
# network.add_layer(12, 1, "sigmoid")
# network.after(show=False)
# data["y"] = data1["y"]
# x_train, y_train, x_valid, y_valid, x_test, y_test = splitting_data(data)
# network =training(data,x_test,y_test ,network =network,batch_size=64,range_i =1)

data = pd.read_csv(r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\training_data.csv",delimiter=";")

nn = NNetwork.create_instance(alfa=0.000001)


nn.after(show=True)
x_train, y_train, x_valid, y_valid, x_test, y_test = splitting_data(data)
training(data,x_test,y_test ,network =nn,batch_size=64,range_i =1)


X = data.iloc[:,:-1].values
Y = data.iloc[:,-1].values




nn.after(show=True)
predictions= []
print(nn.perceptron(X,Y))
for x_point,y_point in zip(X,Y):
    out = nn.pred(x_point)[0]
    y_pred = 1 if out>=0.51 else 0
    #print(out, y_pred,y_point)
    predictions.append(y_pred)
print(nn.confusion_matrix(predictions,Y.tolist()))


# data = data_preprocessing([1, 4, 6, 7, 8, 12],[2, 9, 10, 11])
# x_train, y_train, x_valid, y_valid, x_test, y_test = splitting_data(data)
# print(x_train[1])
# #wiedzieć jak zbudować sieć datsets w pytorch, transfer learning
#
# def main_training_multi_layers(data):
#     processes = []
#     for i in range(5):
#         process = multiprocessing.Process(target=training, args=(x_train, y_train, x_valid, y_valid, x_test, y_test, i + 1,0.0018,64,"RMSprop"))
#         process.start()
#         processes.append(process)
#         print("process szkolenie")
#
#     for process in processes:
#         process.start()
#
# if __name__ == "__main__":
#     # Example usage:
#     main_training_multi_layers(data)
#     print("its over")
