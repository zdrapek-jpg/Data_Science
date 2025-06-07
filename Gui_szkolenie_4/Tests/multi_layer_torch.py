import random

import pandas as pd
from torch.nn import Sequential, Linear,CrossEntropyLoss,ReLU
from torch import from_numpy,no_grad,argmax
from torch.optim import SGD,Adam
from pandas import read_csv

linear_model = Sequential(Linear(4,4),
                          ReLU(),
    Linear(4,3),
                          )

data_load=  read_csv(r"C:\Program Files\Pulpit\Data_science\Zbiory\iris_orig.csv",delimiter=",")
data_load.columns=["1","2","3","4","y"]
from Data.Transformers import Transformations,StandardizationType
from Data.One_hot_Encoder import OneHotEncoder
from Data.SPLIT_test_valid_train import SplitData


##  obsługa y i multi klas (3)
y_for_one_hot = data_load.loc[:,"y"]
unique  = sorted(pd.unique(data_load["y"]),reverse=True)

one_hot = OneHotEncoder()
one_hot.label_encoder_keys(y_for_one_hot,[unique])
y =one_hot.code_y_for_network(y_for_one_hot).values

### obsługa x tylko standaryzacja
x = data_load.iloc[:,:-1]
norm = Transformations(std_type=StandardizationType.Z_SCORE)
X =norm.standarization_of_data(x).values


splitter =SplitData()
splitter.set(train=0.8,valid=0.1,groups=5)
## indexy 5 grup które trenujemy
splits = splitter.split_in_groups(X,groups=5,size=0.9)
### z tego wybieramy splita na którym jest trenowany zbiór
grupy= splitter.get_in_order(splits)



optimizer = Adam(linear_model.parameters(),lr=0.23)
criterion = CrossEntropyLoss()
data = splitter.merge(X,y)

losses= []
X = from_numpy(X).float()
y = from_numpy(y).long()

for i in range(150):
    # train_idx, valid_idx = random.sample(grupy, 2)
    # x_train,y_train =X[splits[train_idx][0]], y[splits[train_idx][0]]
    # x_test,y_test = X[splits[train_idx][1]],y[splits[train_idx][1]]
    x_train, y_train,_,_,_,_ = splitter.split_data(data)
    x_train = from_numpy(x_train).float()
    y_train = from_numpy(y_train).long()

    pred =linear_model(x_train)
    loss = criterion(pred,y_train.reshape(-1,))
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%10==0:
        print(loss.item())

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6,6))
plt.plot(range(150),losses,c="green",linewidth =3)
plt.title("wykres straty")
plt.show()
with no_grad():
    pred =linear_model(X)
    loss = criterion(pred,y.reshape(-1,))
    from NeuralNetwork.Network_single_class1 import NNetwork
    predctions = [argmax(pred_).item() for pred_ in pred]
    print(NNetwork.confusion_matrix(predctions,y.reshape(-1,).tolist()))





















