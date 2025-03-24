#testy czy struktura warstwy neuronowej działa poprawnie
import pandas as pd
import numpy as np

from Data.Multiple_points import multiply
from NeuralNetwork.One_Layer import LayerFunctions
from Data.SPLIT_test_valid_train import SplitData
from Data.Transformers import Transformations

data_piersi = pd.read_csv("wdbc.csv")
x = data_piersi.iloc[:,2:]
obj =Normalization()
x_normalized = obj.normalizacja(x)
x =np.array(x_normalized)
y = data_piersi.iloc[:,1]
#print(x)

y, _ = pd.factorize(y)
#print(y)
data_frame_piersi = pd.DataFrame(x)
data_frame_piersi["label"]=y
print(data_frame_piersi)

### obiekt do podziału z x i y w sobie razem jako jeden obiekt
#to co zwracamy jest już podzielne na x i y walidacyjne, testowe i treningowe


SplitData.set(train=0.40,valid= 0.40,test =0.20)
x_train,y_train,x_valid,y_valid,x_test,y_test=SplitData.split_data(data_frame_piersi)
def test_piersi(x_train,y_train,x_validate,y_validate,x_test,y_test):
    #4X4 4X3

    network = NNetwork(epoki=10, alpha=0.1)


    network.train_backward(x_train, y_train, x_validate, y_validate)
    network.epoki=15
    network.alph=0.001
    network.train_backward(x_train, y_train, x_validate, y_validate)


    # listy danych predykcji
    out = network.perceptron(x_test,y_test)
    print("test accuracy:",out )
    net_loss = network.loss
    net_acc = network.accuracy

    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(111)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (Network and Linear)")

    ax1.plot(net_loss, label="Net Loss", color=(0.5, 0.3, 0.1, 0.9))  # red green blue alpha
    ax1.plot(net_acc, label="Net Accuracy", color=(0.6, 0.4, 0.1, 0.4))
    ax1.set_ylim([0, 1.3])


    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='upper right')

    plt.title("Loss and Accuracy over Epochs")
    plt.show()
    print("\n",network.after())

print(test_piersi(x_train, y_train, x_valid, y_valid, x_test, y_test))