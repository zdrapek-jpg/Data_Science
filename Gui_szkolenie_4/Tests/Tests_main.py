import numpy as np
import pandas as pd
from Data.Multiple_points import multiply
from Data.SPLIT_test_valid_train import SplitData
from Data.Transformers import Transformations
from Data.Transformers import StandardizationType

x =[[1,0,0,0],[1,0.1,0,0],[0,0,1,1],[0,0,1,1],[1,0.1,0.1,0]]
y = [0,0,1,1,0]
#create pandas data frame with x,y
data_frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(len(x[0])) ])
data_frame["y"] = y

# multipy data points
# poiwieleone 8 razy i threshold jest na poziomie +|- 0.01
data =multiply(data_frame,4,0.2)
y = data.loc[:,'y']
x = data.iloc[:,:-1]
#print(x)
#normalizacja danych metodą min, max
norma = Transformations(x,std_type=StandardizationType.NORMALIZATION)
x = norma.standarization_of_data(x)

from NeuralNetwork.Network_single_class import NNetwork
x = x.values
y = y.values
network = NNetwork(alpha=0.1,epoki=11)
network.train(x,y,x,y)
network.perceptron(x,y)
print("acc: ",network.train_accuracy)
print(network.loss)
print(network.after())


print()
print()

## sposób wyliczania gradientu do tyłu w warstwach ukrytych czyli 1 i 2
val = x[0]
wagi = network.LineTwo.wagi
print(wagi)
biasy = np.array([1,1,0],dtype='float64')
gradient = np.array([0.0,0.0,1.0])
output = ([0.0,0.9,0.5])
pochodna_akt = np.array([0.5, 0.1, 0.8])

alfa = np.array([1])
gradient = np.dot(wagi.T,gradient)
gradient *=pochodna_akt
wu = np.outer(gradient,alfa)
print(wu)
biasy -=alfa*gradient
wagi -= alfa*wu*output
print(wagi)




def test_piersi(x_train,y_train,x_validate,y_validate,x_test,y_test):
    SplitData.set(train=0.40, valid=0.40, test=0.20)
    x_train, y_train, x_valid, y_valid, x_test, y_test = SplitData.split_data()
    #4X4 4X3

    network = NNetwork(epoki=10, alpha=0.1)


    network.train(x_train, y_train, x_validate, y_validate)
    network.epoki=15
    network.alpha=0.05
    network.train(x_train, y_train, x_validate, y_validate)



    # listy danych predykcji
    out = network.perceptron(x_test,y_test)
    print("test accuracy:",out )
    net_loss = network.loss
    net_acc = network.train_accuracy

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

#print(test_piersi(x_train, y_train, x_valid, y_valid, x_test, y_test))

