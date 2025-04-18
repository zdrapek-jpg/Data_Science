import numpy as np
import pandas as pd
#
# data_1 = pd.read_csv(r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\bank-full.csv",delimiter=";")
# data_1["balance"] = data_1["balance"].replace("yes",1000)
# data_1["balance"] = np.where(data_1["balance"]>= 900, data_1["balance"] * 3, data_1["balance"])
# data_1["balance"] = data_1["balance"].astype(int)
# data_1["balance"] = pd.to_numeric(data_1["balance"], errors="coerce")
# #print(data_1["loan"].unique())
#
# # modyfikacja danych
# data_1.loc[(data_1.age>28) & (data_1.marital=="married"),"y"]= "yes"
# data_1.loc[((data_1.marital=="divorced")& (data_1.age<29)) | ((data_1.balance<=300)& (data_1.loan=="yes")) ,"y"  ] = "no"
# #print(data_1["balance"])
#
#
#
#
# # kolumna ma 2 wartości "yes", "no"
# #za pomocą numpy zamieniamy wartośći na 0/1
# #na koncu rzutujemy dane na tak lub nie
# # new_y =np.asarray(np.where(y_1=="yes",1,0))
#
# data_2 = pd.read_csv(r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\DataSet_for_ANN-checkpoint.csv")
#
# y_2 = data_2.loc[:,"Exited"]
#
# #wartości dla danych z zbioru 1 i 2   ograniczamy się do wartości od 0 do 1000 wierszy
# x1  =data_1.iloc[:1001,[0,1,2,3,5,7]]
# x2 = data_2.iloc[:1001,[2,3,4,5,7,10,11,12]]#10
#
# merged_data = pd.concat([x2,x1],axis=1)
#
#
# y_1 = data_1.loc[:1001,"y"]
# merged_data["y"]= y_1
#
# merged_data.loc[(merged_data["loan"]=="no")&(merged_data["IsActiveMember"]==0) & (merged_data["job"]=="blue-collar"),"y"]="no"
#
# merged_data.rename(columns={merged_data.columns[2]: "Country"}, inplace=True)
#
# merged_data.to_csv(path_or_buf=r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\new_data.csv",sep=";",columns=merged_data.keys(),index=False)
#
#
# val = merged_data["y"].unique()
# for value in val:
#     x =merged_data["y"].value_counts()[value]

#### tworzenie danych dotyczących uczenia modelu

data_set =pd.DataFrame()
for k in for_one_hot.keys():
    data = for_one_hot.loc[:,k].values

    one_hot = OneHotEncoder(data)
    one_hot.code_keys(data)
    print(one_hot.label_code)
    print(one_hot.code_keys(data))
    data_modified = one_hot.code_y_for_network(data)
    print(data_modified)
    data_set =pd.concat((data_set,data_modified),axis=1)

data_set =pd.concat((data_set,trained_data),axis=1)
y.replace(("yes", "no"), (1, 0), inplace=True)
data_set["y"]=y
print(data_set)
data_set.to_csv(path_or_buf=r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\training_data.csv",sep=";",columns=data_set.keys(),index=False)







