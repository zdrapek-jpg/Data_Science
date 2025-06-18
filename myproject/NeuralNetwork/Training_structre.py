import numpy as np

from NeuralNetwork.Network_single_class1 import NNetwork
from Data.Decorator_time_logging import log_execution_time
@log_execution_time
def splitting_data(data_transformed,train_size=0.2,valid_size=0.4,test_size=0.4):
    from Data.SPLIT_test_valid_train import SplitData
    Split = SplitData()
    # ustawiamy % danych na zbiór
    Split.set(train=train_size, valid=valid_size, test=test_size)
    # podział
    x_train, y_train, x_valid, y_valid, x_test, y_test = Split.split_data(data_transformed)

    return x_train, y_train, x_valid, y_valid, x_test, y_test
import random
random.seed(32)
@log_execution_time
def training(data,x_test,y_test,network=None,batch_size=32,range_i=""):


    network.train_mini_batch(data,batch_size)
    net_loss = network.train_loss
    net_acc = network.train_accuracy
    valid_loss = network.valid_loss
    valid_accuracy=network.valid_accuracy
    test_acc, test_loss = network.perceptron(x_test, y_test)
    print("train loss:  ",net_loss[-1],    " train acc: ", net_acc[-1],)
    print("valid loss:  ",valid_loss[-1], "  valid acc: ",valid_accuracy[-1] )
    print("test loss:   ", test_loss,           "  test acc   ",test_acc )
    path= f"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\model{range_i}"+".json"
    #network.write_model(path=path)

    from NeuralNetwork.Show_results import show_training_process

    show_training_process(network.train_accuracy,network.train_loss,network.valid_accuracy,network.valid_loss,test_acc,test_loss,index =range_i)
    data_x= data.iloc[:,:-1].values
    y = data.iloc[:,-1].values
    skutecznosc, strata = network.perceptron(data_x, y)
    #network.after(show=True)

    #####  TO DOKONCZYĆ BO FUNCKJA WRACA ARRAY (31,1) MUS BYĆ (31,) BO PRED ZWRUCI (1,12)
    print("test accuracy: ", skutecznosc, " test loss: ", strata)
    predictions = []

    #print(network.perceptron(data_x, y))
    for x_point, y_point in zip(data_x, y):
        out = network.pred(x_point)[0]
        y_pred = 1 if out >= 0.51 else 0
        # print(out, y_pred,y_point)
        predictions.append(y_pred)
    print(network.confusion_matrix(predictions, y.tolist()))
    user = input("save model?")
    if user == "y":
        print("model zapisany")
        network.write_model("C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\model.json")
        instance = network.create_instance()
        print(instance.perceptron(data_x, y))
    return network

#
#
# network = NNetwork(epoki=80, alpha=0.008, optimizer="RMSprop",
#                    gradients="mini-batch")  # optimizer="momentum",gradients="batch"
# network.add_layer(31, 12, "relu",seed_for_all=32)
# network.add_layer(12, 12, "relu",seed_for_all=32)
# network.add_layer(12, 1, "sigmoid",seed_for_all=32)
# # network =training(data,x_test,y_test ,network =network,batch_size=64,range_i =1)
# from pandas import read_csv
# data = read_csv(r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\training_data.csv",delimiter=";")
# x_train, y_train, x_valid, y_valid, x_test, y_test = splitting_data(data,train_size =0.6,valid_size=0.2,test_size=0.2)
# #training(data,x_test,y_test ,network =network,batch_size=32,range_i =1)
#
# import multiprocessing
# def main_training_multi_layers(data):
#     processes = []
#     for i in range(3):
#         process = multiprocessing.Process(target=training, args=(data,x_test,y_test,network,32,i))
#         processes.append(process)
#         print("process szkolenie")
#     for process in processes:
#         process.start()
#
# if __name__ == "__main__":
#     # Example usage:
#     main_training_multi_layers(data)
#     print("its over")