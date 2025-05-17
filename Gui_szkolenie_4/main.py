import pandas as pd

from NeuralNetwork.Training_structre import training
from NeuralNetwork.getData import data_preprocessing
from NeuralNetwork.Network_single_class1 import NNetwork

import multiprocessing
import time
import os
# załadowanie i przetworzenie danych do modelu1

# stworzenie modelu i szkolenie go na danych data
data = pd.read_csv(r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\training_data.csv",delimiter=";")
#print(data.columns)
training(data,"")
# x = data.iloc[:,:-1].values
# y =data.iloc[:,-1].values
# nn = NNetwork.create_instance()
# nn.after()
# acc,blod =nn.perceptron(x,y)
# print(acc," ",blod)

def main_training_multi_layers(data):
    processes = []
    for i in range(5):
        process = multiprocessing.Process(target=training, args=(data, i + 1))
        process.start()
        processes.append(process)
        print("process szkolenie")
    for process in processes:
        process.start()



if __name__ == "__main__":
    # Example usage:
    data = data_preprocessing()
    main_training_multi_layers(data)
    print("its over")

