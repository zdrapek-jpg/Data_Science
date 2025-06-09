import pandas as pd

import NeuralNetwork.Network_single_class1
from basic_functions import *
path1= r"C:\Program Files\Pulpit\Data_science\Zbiory\iris_orig.csv"
path2 = r"C:\Program Files\Pulpit\Data_science\Zbiory\heart.csv"
path3 = r"C:\Program Files\Pulpit\Data_science\Zbiory\DataSet_for_ANN-checkpoint.csv"
path4 = r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\bank-full.csv"
# data = read_data(path1,delim=",",columns=["a1","a2","a3","a4","y"])
data = read_data(path4,delim=";")

y = data.iloc[:,-1]
print(data.head(5))
# 0 kolumna z-socre
# 1 kolumna one-hot one_hot_encoder
# 2 kolumna one-hot one_hot_encoder
# 3 kolumna one-hot one_hot_encoder
# 4 kolumna one-hot one_hot_encoder
# 5 kolumna z-score
# 6 kolumna one-hot one_hot_encoder
# 7 kolumna one-hot one_hot_encoder
# 8 kolumna one-hot one_hot_encoder
# 9 kolumna one-hot one_hot_encoder
# 10kolumna z-score
# 11kolumna label_encoder -> z-score
# 12kolumna z-score
# 13kolumna z-score
# 14kolumna z-score
# 15kolumna z-score
# 16kolumna one-hot one_hot_encoder
# # 17kolumna one-hot one_hot_encoder

for_zsocre = data.iloc[:,[0,5,9,11,12,13,14]]
for_one_hot = data.iloc[:,[1,2,3,4,6,7,8,10,15,16]]
for_label_encoder = data.iloc[:,-1]
# without_change = data.iloc[:,[7,8]]z
# for col in list(data.columns):
#     print(data[col].unique())
order = data["y"].unique()
# #
x1,std =standarization(StandardizationType.Z_SCORE,for_zsocre)
x2,oneh= one_hot_encoder(for_one_hot)
y,label = label_encoder(for_label_encoder,[order])
# # #x2,std2 = standarization(StandardizationType.NORMALIZATION,for_normalization)
x = pd.concat((x1,x2),axis=1)
data = pd.concat((x,y),axis=1)
print(data.shape)
# data.to_csv("data_transformed.csv",columns=data.columns,sep=";")
#data = pd.read_csv(r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\torch_z_sets\data_transformed.csv",delimiter=";")

x = data.iloc[:,:-1]
y = data.iloc[:,-1]



### konkretne przekazanie danych do modelu  i całość jest zautomatyzowana
training(15,x.values,y.values,lr=0.0001,min_acc=0.89,below_error=0.1166,size=0.9)

x= x.values
y= y.values
network_retrained = Net()
network_retrained.load_state_dict(load(r'bank_30k_model5.pt'))
network_retrained.eval()

criterion = MSELoss(reduction="mean")

preds =network_retrained.forward(from_numpy(x).float())
preds = clamp(preds,min=0.0,max=1.0)
loss = criterion(preds, from_numpy(y).reshape(-1,1).long())
print("loss: ",loss.item())
preds = preds.reshape(-1,).tolist()
preds = list(map(round, preds))
y = y.reshape(-1,).tolist()
print(NeuralNetwork.Network_single_class1.NNetwork.confusion_matrix(preds,y))
accuracy = sum([1 for y1,y2 in zip(preds,y) if round(y1)==y2])/len(y)
print("accuracy",accuracy)

# # #13 wierszy dla heart
# # new_data_point = [67,1,100, 140,1,1, 1,1,1,3]
# # regular = [0,0,1]
# #
# # for_forward =standarization_one_point(std,new_data_point)
# # print(for_forward)
# # for_forward = for_forward+regular
# # print(for_forward)
# # x =network_retrained.forward(Tensor(for_forward))
# # print(x)
## 12 wierszy dla bankNN
# new_data_point = pd.DataFrame([[1035,"Spain","Male",32,2,1,2,1,0,23413.34]],columns=data.columns)
# for_zsocre = new_data_point.iloc[:,[0,3,4,5,6,9]]
# for_one_hot = new_data_point.iloc[:,1]
# for_label_encoder = new_data_point.iloc[:,2]
# without_change = new_data_point.iloc[:,[7,8]]
# # for col in list(data.columns):
# #     print(data[col].unique())
# order = data["Gender"].unique()
# x1 = standarization_one_point(std,for_zsocre.values.tolist()[0])
# x2 = code_one_hot_point(for_one_hot,oneh)
# x3 = code_one_hot_point(for_label_encoder,label)
# x1 = pd.DataFrame([x1])
# for_network = pd.concat((x1,x2,x3,without_change),axis=1)
# print(for_network)
# model = Net()
# model.load_state_dict(load(r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\torch_z_sets\banknn_model4.pt"))
# #
# print(model.forward(from_numpy(for_network.values).float()))