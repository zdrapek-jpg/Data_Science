##### DATA   :  Diamonds csv
import random

from torch.nn import Sequential,Linear,Sigmoid,MSELoss,ReLU
from torch.optim import SGD,Adam,RMSprop
from torch import from_numpy, no_grad
from pandas import read_csv,concat
from Data.Transformers import Transformations,StandardizationType
data = read_csv(r"C:\Program Files\Pulpit\Data_science\Zbiory\heart.csv",delimiter=",")
print(data.info())
y = data.loc[:,"target"]
columns = data.columns
### kolumny które zostają 1,2,5,6,8,10,11,12
x = data.iloc[:,[0,3,4,7,9]]
old =data.iloc[:,[1,2,5,6,8,10,11,12]]
norm = Transformations(std_type=StandardizationType.NORMALIZATION)
x = norm.standarization_of_data(x)

X = concat((x,old),axis=1)
print(X)
print(y)
from Data.SPLIT_test_valid_train import SplitData
splitter = SplitData()
splitter.set(0.8,0.1,0.1,4)
groups = splitter.split_in_groups(X,3,size=0.9)
groups_idx = splitter.get_in_order(groups)
X =from_numpy(X.values).float()
y = from_numpy(y.values).float()

linear_model = Sequential(Linear(13,13,bias=True),
                          ReLU(),
                          Linear(13,1,bias=True),
                          Sigmoid())

optimizer = Adam(linear_model.parameters(),lr=0.04)
criterion = MSELoss()
train_losses= []
valid_loss  = []
test_loss   = []
for i in range(100):
    train_idx,valid_idx = random.sample(groups_idx,2)
    x_train = X[groups[train_idx][0]]
    y_train = y[groups[train_idx][0]]


    pred = linear_model(x_train)
    loss = criterion(pred,y_train.reshape(-1,1))
    train_losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%10==0:
        print("Loss: ",loss.item() )
    with no_grad():
        pred = linear_model(X)
        x_train = X[groups[valid_idx][1]]
        y_train = y[groups[valid_idx][1]]
        from NeuralNetwork.Network_single_class1 import NNetwork

        predictions = [round(pred_.item()) for pred_ in pred]
        loss = criterion(pred,y.reshape(-1,1))
        valid_loss.append(loss.item())

        print(NNetwork.confusion_matrix(predictions, y.tolist()))

with no_grad():
    pred = linear_model(X)
    from NeuralNetwork.Network_single_class1 import NNetwork
    predictions = [round(pred_.item()) for pred_ in pred]

    print(NNetwork.confusion_matrix(predictions,y.tolist()))

import matplotlib.pyplot as plt

figure = plt.figure(figsize=(6,6))
ax1 = figure.add_subplot(111)
ax1.set_xlabel("Epochs")
ax1.set_ylabel("loss network ")

ax1.plot(train_losses, label="train Loss", color=(0.5, 0.3, 0.1, 0.9))  # red green blue alpha
ax1.plot(valid_loss, label="valid Loss", color=(0.6, 0.4, 0.1, 0.4))
plt.show()




