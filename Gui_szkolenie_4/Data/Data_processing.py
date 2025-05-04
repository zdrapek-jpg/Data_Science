import numpy as np
import pandas as pd

from NeuralNetwork.getData import load_data

data_1 = load_data(file_path=r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\bank-full.csv")

data_1["balance"] = data_1["balance"].replace("yes",5000)
data_1["balance"] = np.where(data_1["balance"]>= 1000, data_1["balance"] * 3, data_1["balance"])
data_1["balance"] = data_1["balance"].astype(int)
data_1["balance"] = pd.to_numeric(data_1["balance"], errors="coerce")
#print(data_1["loan"].unique())

# modyfikacja danych
data_1.loc[(data_1.age>21) & (data_1.marital=="married"),"y"]= "no"
data_1.loc[(data_1.age>23) & (data_1.balance<=4000) & (data_1.marital!="married"),"y"]= "no"
data_1.loc[((data_1.marital=="divorced") |( data_1.marital=="single")) &((data_1.balance<=4000)& (data_1.loan=="yes")) ,"y"  ] = "no"
data_1.loc[(data_1.marital=="married")| ((data_1.balance>5400)& (data_1.loan!="yes")) ,"y"  ] = "yes"
data_1.loc[(data_1.marital=="single") &  (data_1.loan=="yes") ,"y"  ] = "no"
data_1.loc[(data_1.loan=="yes") ,"y"  ] = "no"
data_1.loc[(data_1.balance>=4000) & (data_1.loan!="yes"),"y"] = "yes"


#print(data_1["balance"])




# kolumna ma 2 wartości "yes", "no"
#za pomocą numpy zamieniamy wartośći na 0/1
#na koncu rzutujemy dane na tak lub nie
# new_y =np.asarray(np.where(y_1=="yes",1,0))

data_2 = load_data(file_path=r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\DataSet_for_ANN-checkpoint.csv")

y_2 = data_2.loc[:,"Exited"]

#wartości dla danych z zbioru 1 i 2   ograniczamy się do wartości od 0 do 1000 wierszy
x1  =data_1.iloc[:1001,[0,1,2,3,5,7]]
x2 = data_2.iloc[:1001,[2,3,4,5,7,10,11,12]]#10

merged_data = pd.concat([x2,x1],axis=1)


y_1 = data_1.loc[:1001,"y"]
merged_data["y"]= y_1

merged_data.loc[(merged_data["loan"]=="no")&(merged_data["IsActiveMember"]==0) & (merged_data["job"]=="blue-collar"),"y"]="no"

merged_data.rename(columns={merged_data.columns[2]: "Country"}, inplace=True)

merged_data.to_csv(path_or_buf=r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\new_data.csv",sep=";",columns=merged_data.keys(),index=False)


val = merged_data["y"].unique()
print(val)
for value in val:
    x =merged_data["y"].value_counts()[value]
    print(x)

