from torch.nn import functional,Linear,Module,Sigmoid
from torch import manual_seed,from_numpy,no_grad,save,argmax,load,Tensor,clamp
from torch.optim import Adam,RMSprop,Adagrad,SGD,Adadelta,ASGD
from torch.nn  import CrossEntropyLoss,MSELoss
import numpy as np
import random
random.seed(42)
manual_seed(41)

class Net(Module):
    def __init__(self):

        super().__init__()

        self.l1 = Linear(90,90,bias=True)
        self.l2 = Linear(90,30,bias=True)
        self.output = Linear(30,1,bias=True)


    def forward(self, data):

        l1output = functional.elu(self.l1(data))
        l2output = functional.elu(self.l2(l1output))
        out = self.output(l2output)
        return functional.sigmoid(out)

from Data.Transformers import StandardizationType, Transformations
from pandas.core.frame import DataFrame,Series

def read_data(path,delim,columns=None):
    from pandas import read_csv
    if columns:
        return  read_csv(path,delimiter=delim,names=columns)
    else :  return read_csv(path,delimiter=delim)


def standarization(std_type:StandardizationType,data:DataFrame)->tuple[DataFrame,Transformations]:
    std = Transformations(std_type=std_type)
    return std.standarization_of_data(data),std

def standarization_one_point(model:Transformations,point:list):
    return model.standarization_one_point(point)


from Data.One_hot_Encoder import OneHotEncoder

def one_hot_encoder(data:DataFrame)-> tuple[DataFrame, OneHotEncoder]:
    one_hot = OneHotEncoder()
    one_hot.code_keys(data)
    return one_hot.code_y_for_network(data),one_hot

def label_encoder(data:DataFrame,order:list)->tuple[DataFrame,OneHotEncoder]:
    onehot = OneHotEncoder()
    onehot.label_encoder_keys(data,order)
    return onehot.code_y_for_network(data),onehot

def code_one_hot_point(data:Series,model:OneHotEncoder):
    return model.code_y_for_network(data)
from Data.SPLIT_test_valid_train import SplitData
def split_data_return_one_fold(x,y,size):

    splitter = SplitData()
    splitter.set(groups=2)
    return splitter.split_in_groups(x ,size= size)       ### 5 group tasowanych każda grupa inna, każda zawiera treningowy i testowy zbiór 0.7 ; 0.3





def training(epochs,x,y,lr=0.01,size=0.6,min_acc=0.99,below_error=0.045):
    x, y = from_numpy(x).float(), from_numpy(y).float()
    criterion = MSELoss(reduction="mean")
    for_one_optimizer=[]

    for index in range(6):
        network = Net()
        #network.load_state_dict(load(r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\torch_z_sets\bank_30k_model0.pt"))

        optimizers = [Adam(network.parameters(),lr=lr),
                      RMSprop(network.parameters(),lr=lr),
                      SGD(network.parameters(),lr=lr,momentum=0.94),
                      Adagrad(network.parameters(),lr=lr),
                      Adadelta(network.parameters(),lr=lr)
                      ]
        train_losses=[]
        for j in range(epochs):
            flag =False
            ## grupy danych
            for_epoch_loss=[]
            groups =split_data_return_one_fold(x,y,size=size)
            for i in range(len(groups)):
                train_x,train_y = x[groups[i][0]],y[groups[i][0]]
                valid_x,valid_y = x[groups[i][1]],y[groups[i][1]]
                prediction =network.forward(train_x)
                prediction= clamp(prediction,min=0.0,max=1.0)
                #print(prediction)
                loss =criterion(prediction,train_y.reshape(-1,1))
                for_epoch_loss.append(loss)
                optimizers[i].zero_grad()
                loss.backward()
                optimizers[i].step()
                with no_grad():
                    pred = network.forward(valid_x)
                    clamp(pred,min=0.0, max=1.0)
                    valid_loss = criterion(pred, valid_y.reshape(-1,1))

                    acc = sum([1 for y1, y2 in zip(pred.reshape(-1,).tolist(), valid_y.tolist()) if
                               round(y1) == y2]) / (len(pred))
                    print(valid_loss.item()," acc: ",acc)

                    if valid_loss.item() <= below_error and acc >= min_acc:
                        pred = network.forward(x)
                        pred = clamp(pred, min=0.0, max=1.0)
                        loss = criterion(pred, y.reshape(-1,1))

                        acc = sum([1 for y1, y2 in zip(pred.reshape(-1,).tolist(), y.tolist()) if
                                   round(y1) == y2]) / (len(pred))
                        print("skuteczność",acc, " strata: ",loss)
                        if loss<=below_error+0.013 and acc>=min_acc-0.05:
                            save(network.state_dict(), f"bank_30k_model{index}.pt")
                            flag = True
                            break


            train_losses.append((sum(for_epoch_loss) / len(for_epoch_loss)).item())

            if j%30==0:
                print(train_losses[-1])
            if flag:
                break

        for_one_optimizer.append(train_losses)


    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from numpy import linspace
    color = cm.tab10(linspace(0,1,len(for_one_optimizer)))
    nazwy = ["Adam","RMSprop","SGD","Adagrad","Adadelta","ASGD"]
    fig = plt.figure(figsize=(8, 6))
    ax1 = plt.subplot(111)
    for i,strata_dla_opt in enumerate(for_one_optimizer):
        ax1.plot(range(len(strata_dla_opt)), strata_dla_opt,label=nazwy[i], c=color[i], linewidth=1.5)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.show()
