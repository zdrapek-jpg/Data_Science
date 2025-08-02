import time
from torchvision.datasets import MNIST
from torchvision import transforms
from torch import nn, save,optim
from torch.optim import lr_scheduler
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset

transform = transforms.ToTensor()
Train_data = MNIST(root= "/mnist" ,train=True,download=False,transform=transform)
test_data = MNIST(root= "/mnist" ,train=False,download=False,transform=transform)
#print(Train_data)
#print(test_data)
from sklearn.model_selection import ShuffleSplit
splitter = ShuffleSplit(n_splits=1,train_size=0.8,test_size=0.2)

for train_idx, test_idx in splitter.split(Train_data):
    train_subset = Subset(Train_data, train_idx)
    test_subset = Subset(Train_data, test_idx)

train_loader = DataLoader(train_subset,batch_size=16,shuffle=True)
test_loader =  DataLoader(test_subset,batch_size=16, shuffle=False)
    # input, output, kernal , number
conv1 = nn.Conv2d(1,6,3,1)
conv2 = nn.Conv2d(6,16,3,1)
single_data = Train_data[100][0]
x = single_data.view(1,1,28,28)
#print(x)

##  konwolucja
conv_1 = F.relu(conv1(x))
print("konwolucja ",conv_1.shape)

## pooling
x = F.max_pool2d(conv_1,2,2)
print("pooling ",x.shape)

new_x = F.relu(conv2(x))
print("kolejna konwolucja ",new_x.shape)
## second convolution


x = F.max_pool2d(new_x,2,2)
print("kolejny pooling : ", x.shape)

class ConvolutionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,3,1)
        self.conv2= nn.Conv2d(6,16,3,1)
        self.NeuronLine =  nn.Linear(5*5*16,120)
        self.NeuronLine2 = nn.Linear(120,60)
        self.NeuronLine3 = nn.Linear(60,10)

    def forward(self,X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)
        X = X.view(-1,16*5*5)
        X = F.relu(self.NeuronLine(X))
        X = F.relu(self.NeuronLine2(X))
        X = F.relu(self.NeuronLine3(X))
        return F.log_softmax(X,dim=1)
#torch.manual_seed(41)

model = ConvolutionModel()
lr=0.0005
from torch import optim
start= time.time()
print(start)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=lr)
train_loss = []
test_loss = []
train_correct = []
test_correct = []


for epoch in range(4):
    model.train()
    scheduler =lr_scheduler.StepLR(optimizer,gamma=0.8,step_size=1)
    acc= 0
    ls = 0
    for b, (X_train, Y_train) in enumerate(train_loader):

        y_pred = model(X_train)
        loss = criterion(y_pred, Y_train)
        ls+=loss
        predicted = torch.max(y_pred.data, 1)[1]
        correct = (predicted == Y_train).sum().item()
        acc += correct / len(Y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Append for plotting

        if b%300==0:
            print(f"Epoch {epoch}, Batch {b}, Loss: {loss.item():.4f}, Accuracy: {acc/len(train_loader):.4f} , {optimizer.param_groups[0]['lr']}")
    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]
    print("Epoch %d: SGD lr %.4f -> %.4f" % (epoch, before_lr, after_lr))
    train_correct.append(acc/len(train_loader))
    train_loss.append((ls/len(train_loader)).item())


    # Evaluation
    model.eval()

    test_cor=0
    test_l=0

    with torch.no_grad():
        for batch,(X_test, Y_test) in enumerate(test_loader):
            y_val = model(X_test)
            loss = criterion(y_val, Y_test)
            predicted = torch.max(y_val.data, 1)[1]
            test_cor+=(predicted == Y_test).sum().item()/len(Y_test)
            test_l+=(loss.item())
        if test_cor/len(test_loader)>=0.99 and test_l/len(test_loader)<=0.1:
            save(model.state_dict(), "model_mnist.pt")
            print(f"model learned with acc:{test_cor/len(test_loader)}, loss:{test_l/len(test_loader)}")
            break
        test_correct.append(test_cor)
        test_loss.append(test_l/len(test_loader))


save(model.state_dict(),"model_mnist.pt")
print(test_correct)
print(test_loss)

print(train_loss,type(train_loss))
print(train_correct,type(train_correct))
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(train_correct, label="Batch Loss", color='blue')
plt.plot(train_loss, label="Batch Accuracy", color='green')
plt.title("Loss and Accuracy per Batch")
plt.xlabel("Batch")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
