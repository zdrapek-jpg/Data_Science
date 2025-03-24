# Podział danych transformacja oraz uczenie sieci neuronowej
from Data.Transformers import Transformations
from One_Layer import LayerFunctions
import pandas as pd


from Data.SPLIT_test_valid_train import SplitData
from Network_single_class import *

perc = layer_functions(len_data=2, wyjscie_ilosc=3, optimizer="backpropagation", activation_layer="sigmoid")


def activationtest(data):
    #to jest wyjście z wartwy neuronowej
    output  =perc.activation(data)


    # to jest jak ma wyjść klasa
    perc.activation_layer="softmax"
    out2 = perc.activation(output)
    perc.activation_layer="Elu"
    out3 = perc.activation(data)
    return output,out2,out3


data = [np.array([1,1,1]),np.array([-1,-1,-2]),np.array([1,2,3])]
# for d in data:
#     print(activationtest(d))
# print("\n\n")

def testinitializeParamenters(col,row):
    print(f"wejscia{col},wyjscia{row}")
    #inicjaliacja w warstwie jednej klasy wag biasów alf itd:
    perc.len_data = col
    perc.wyjscia_ilosc = row
    perc.random_weights()
    perc.random_bias()
    perc.random_prog()
    perc.random_alfa()

    return perc.wagi,perc.bias,perc.prog,perc.alfa

# for i in range(1,6):
#     if i >5:
#         i-=2
#         print(testinitializeParamenters(i-1, i - 2))
#
#     print(testinitializeParamenters(i, i + 2))




perc.len_data=2
perc.wyjscia_ilosc=5
perc.random_weights()
perc.random_bias()
perc.random_prog()
perc.random_alfa()
perc.activation_layer="softmax"
def test_forward_function(point):
    suma_w =perc.forward(point)
    print("suma wazona neuronów :",suma_w)
    return  perc.activation(suma_w)
#
# for i in range(3):
#     act = test_forward_function([-3,i*2])
#     print(act)


data = [[1,2]]
perc = layer_functions(len_data=2,wyjscie_ilosc=3,activation_layer="sigmoid")
perc2 = layer_functions(len_data=3,wyjscie_ilosc=3,activation_layer="softmax")
perc.start(alfa=0.1)
perc2.start(alfa=0.1)
def test_gradient_dla_1_warstwy(point,y,act =None):
    print("wagi ", perc.wagi)
    print("bias ", perc.bias)
    if act=="sigmoid":
        perc.activation_layer="sigmoid"
    else:
        perc.activation_layer="softmax"
    suma_w1 = perc.forward(point)
    y_pred1 = perc.activation(suma_w1)
    gradient1 = perc.backward(y, y_pred1, point)
    print("gradient wartości ktore idą do poprzedniej warstwy z konkretnych neuronów", gradient1)
    print("wagi ", perc.wagi)
    print("biasy ", perc.bias)

# print(test_gradient_dla_1_warstwy(data, [0, 1, 0],act="sigmoid"))
# print(test_gradient_dla_1_warstwy(data, [0, 1, 0],act="sigmoid"))


def test_gradients_2_warstwy(point,y):
    print("wagi", perc.wagi)
    print("wagi drugie ", perc2.wagi)
    print(perc.bias)
    print(perc2.bias)

    suma_w = perc.forward(point)
    y_pred = perc.activation(suma_w)
    suma_w2 = perc2.forward(y_pred)
    y_pred2 = perc2.activation(suma_w2)
    print("suma wazona neuronów :", suma_w)
    print("predykcja:",y_pred)
    print("soft max:", y_pred2)
    gradient1 = perc2.backward(y,y_pred2,y_pred)
    gradient2 = perc.backward(y_pred =y_pred, point=point,gradient2=gradient1,weigt_forward=perc2.wagi)
    print(gradient2)
    print("gradient wartości ktore idą do poprzedniej warstwy z konkretnych neuronów",gradient1)
    print("wagi",perc.wagi)
    print("wagi drugie ", perc2.wagi)
    print("biasy",perc.bias)
    print("biasy", perc2.bias)


#print(test_gradients_2_warstwy(data,[0,1,0]))


x = [[2.1,4.8,1.6],[4.3,3.1,3.3],[1.3,3.0,1.2],[2.3,1.3,0.5],[3.2,4.6,3.3],[0.4,2.1,3.61],[1.5,1.1,2.3],[0.1,0.2,1.2]]
y = [1,0,1,0,1,2,2,2]
data_frame = pd.DataFrame(x, columns=["x1","x2","x3"])
data_frame["y"] = y

data =multiply(data_frame,40,0.01)
y = data.loc[:,'y']
x = data.iloc[:,:-1]

norma = Normalization()
x = norma.normalizacja(x)
x = np.array(x)
y = np.array(y)

y1= np.array(y)

layer = layer_functions(len_data=3,wyjscie_ilosc=3,activation_layer="softmax")
Net = NNetwork()
# One Hot Encoder
Net.one_hot_encoder(y)
y =Net.one_hot_encoding(y)
def test_forward_on_dataset(data,y):

    print(y)

    layer.start(alfa=0.05)
    for i in range(30):
        for point,y_origin in zip(data,y):
            predykcja =layer.train_forward(point)
            layer.backward(y_origin=y_origin,y_pred=predykcja,point=point)
        print(layer.wagi)
        print(layer.bias)

#print(test_forward_on_dataset(x,y))


def predykcja_na_modelu(x,y):
    odp = []
    for x_point in x:
        output= layer.train_forward(x_point)
        pred = np.argmax(output)
        odp.append(pred)
    return sum([1 if y_pred==y_origin else 0    for y_pred,y_origin in zip(odp,y)])/len(y)

#print(predykcja_na_modelu(x,y1))

######           test on network with 2 layers first sigmoid 3 naurons second softmax 3 neurons 3 labels   ########
network = NNetwork(epoki=30,alpha=0.1)
def init_one_hot(y):
    network.start_one_hot(y)
    print(network.one_hot_encode_y)
    print(network.one_hot_encode)
    print(network.one_hot_encoded)
#print(init_one_hot(y1))

def forward_network(x,y):
    network.train_backward(x,y)
    #print(network.after())
    print(network.perceptron(x,y))

#print(forward_network(x,y1))

from Neuron_2Podejscie import *


####    porównanie skuteczności modeli 1 warstwy do modelu 2 warst na tej samej ilości epok, tej samej alfie i biasach
# wagi w sieci 2 warstwowej będą takie same
# dataset będzie ten same  bez założenia osiągnięcia konkretnej jakości modelu poprostu 20 epok



x = [[2.1,3.8,1.6],[4.5,3.1,3.2],[1.3,2.8,1.2],[2.1,1.0,0.5],[0.2,1.6,0.3],[0.4,2.1,2.61],[2.1,1.7,3.3]]
y = [1,0,1,0,1,2,2]
data_frame = pd.DataFrame(x, columns=["x1","x2","x3"])
data_frame["y"] = y
net = NNetwork()

data =multiply(data_frame,35,0.30)

y = data.loc[:,'y']
x = data.iloc[:,:-1]

norma = Normalization()
x = norma.normalizacja(x)
x = np.array(x)
y = np.array(y)
y1 = y.copy()



## optymalizacja momentum
bias = [0.5,0.5,0.5]
def test_1layer_vs_2layers(x,y):
    # 3X3 i 3X3
    perc =Perceptron_functions2(len_data=3,wyjscie_ilosc=3,epoki=40,bias=bias,optimizer="backpropagation")
    perc.random_weights()
    perc.alfa=0.01
    perc.forward(x,y)
    perc.perceptron(x,perc.wagi)
    # perc.output  to lista predykcji
    print("\n\n")
    network = NNetwork(epoki=40,alpha=0.008)
    network.start_one_hot(y)
    network.train_backward(x,y,x,y)
    # listy danych predykcji
    print(perc.output)
    out =network.perceptron(x,y)
    net_loss =network.loss
    net_acc =network.accuracy
    single_line_loss = perc.loss
    single_line_acc = perc.accuracy

    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(111)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (Network and Linear)")


    ax1.plot(net_loss, label="Net Loss", color=(0.5, 0.3, 0.1, 0.9))  # red green blue alpha
    ax1.plot(net_acc, label="Net Accuracy", color=(0.6, 0.4, 0.1, 0.4))
    ax1.set_ylim([0, 1.3])

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy (Network and Linear)")
    ax2.plot(single_line_loss, label="Linear Loss", color=(0.2, 0.7, 0.4, 0.9))
    ax2.plot(single_line_acc, label="Linear Accuracy", color=(0.1, 0.8, 0.4, 0.5))
    ax2.set_ylim([0, 1.3])
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')


    plt.title("Loss and Accuracy over Epochs")
    plt.show()

    # dla network:
    #loss: 0.152206
    #accuracy: 0.98
    #dla perceptronu
    #loss: 0.276204
    #accuracy: 0.949
# #
#for i in range(2):
#    print(test_1layer_vs_2layers(x,y))

### porzucone testowanie mini batch to dziadostwo nie dziła prendzej wyjde za siostre mateusza !



data_irys = pd.read_csv("iris_orig2.csv", names=["x1", "x2", "x3", "x4", "label"])

# dopisanie kolumny z wartośćią y [0,1] dla 2 kwiatków z przedział€ danych 0-90 wierszy
data_irys['label'], _ = pd.factorize(data_irys["label"])
x = data_irys.iloc[:, :-1]
y= data_irys.loc[:,'label']
obj =Normalization()
x_normalized = obj.normalizacja(x)
x =np.array(x_normalized)

bias = [0.5,0.5,0.5]
def test_irys(x,y):
    #4X4 4X3
    perc = Perceptron_functions2(len_data=4, wyjscie_ilosc=3, epoki=40, bias=bias, optimizer="backpropagation")
    perc.random_weights()
    perc.alfa = 0.01
    perc.forward(x, y)
    perc.perceptron(x, perc.wagi)
    #perc.output  to lista predykcji
    print("\n\n")
    network = NNetwork(epoki=40, alpha=0.001)
    network.start_one_hot(y)
    network.train_backward(x, y,x,y)
    # listy danych predykcji
    print(perc.output)
    out = network.perceptron(x, y)
    net_loss = network.loss
    net_acc = network.accuracy
    single_line_loss = perc.loss
    single_line_acc = perc.accuracy

    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(111)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (Network and Linear)")

    ax1.plot(net_loss, label="Net Loss", color=(0.5, 0.3, 0.1, 0.9))  # red green blue alpha
    ax1.plot(net_acc, label="Net Accuracy", color=(0.6, 0.4, 0.1, 0.4))
    ax1.set_ylim([0, 1.3])

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy (Network and Linear)")
    ax2.plot(single_line_loss, label="Linear Loss", color=(0.2, 0.7, 0.4, 0.9))
    ax2.plot(single_line_acc, label="Linear Accuracy", color=(0.1, 0.8, 0.4, 0.5))
    ax2.set_ylim([0, 1.3])
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.title("Loss and Accuracy over Epochs")
    plt.show()
#for i in range(4):
#    print(test_irys(x,y))


def test_create_batches(data):
    network = NNetwork(epoki=10, alpha=0.1,batche="batch")
    network.start_one_hot(data.iloc[:,-1])
    data["y"] = network.one_hot_encode_y
    x_train,y_train =network.split_on_batches(data,2)
    network.train_batches(data,x_train,y_train,size=3)
    print(x_train,y_train)

#print(test_create_batches(data))



x = [[2.1,3.8,1.6],[4.5,3.1,3.2],[1.3,2.8,1.2],[2.1,1.0,0.5],[0.2,1.6,0.3],[0.4,2.1,2.61],[2.1,1.7,3.3]]
y = [1,0,1,0,1,2,2]
data_frame = pd.DataFrame(x, columns=["x1","x2","x3"])
data_frame["y"] = y

data =multiply(data_frame,5,0.001)
y = data.loc[:,'y']
x = data.iloc[:,:-1]

norma = Normalization()
x = norma.normalizacja(x)
x = np.array(x)
data_frame = pd.DataFrame(x)
network.start_one_hot(y)

data_frame['y']=y
x_train,y_train,x_valid,y_valid,x_test,y_test=SplitData.split_data(data_frame)


def test_learning_on_batches(data_frame,x_train,y_train,x_valid,y_valid,x_test,y_test):
    network = NNetwork(epoki=10, alpha=0.9,batche="batch")
    network.train_batches(data_frame,x_train,y_train,size=15)

    net_loss = network.loss
    net_acc = network.accuracy

    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(111)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax1.plot(net_loss, label="Net Loss", color=(0.5, 0.3, 0.1, 0.9))  # red green blue alpha
    ax1.plot(net_acc, label="Net Accuracy", color=(0.6, 0.4, 0.1, 0.4))
    ax1.set_ylim([0, 1.3])

    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines, labels, loc='upper right')

    plt.title("Loss and Accuracy over Epochs")
    plt.show()
    print("\n", network.after())
    ac =network.perceptron(x,y)
    print("final",ac)
    print(network.loss[-1])
    return ac



test_learning_on_batches(data_frame,x_train,y_train,x_valid,y_valid,x_test,y_test)

