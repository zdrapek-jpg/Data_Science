
import numpy as np
import math
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
# załadowanie danych
diamonds = pd.read_csv(r"C:\Program Files\Pulpit\Data_science\Zbiory\diamonds.csv",delimiter=",")
# churn = pd.read_csv(r"C:\Program Files\Pulpit\Data_science\Zbiory\churn.csv",delimiter=",")
heart = pd.read_csv(r"C:\Program Files\Pulpit\Data_science\Zbiory\heart.csv",delimiter=",")
# churn_bank = pd.read_csv(r"C:\Program Files\Pulpit\Data_science\Zbiory\Telco-Customer-Churn.csv",delimiter=",")
bank_main = pd.read_csv(r"C:\Program Files\Pulpit\Data_science\Zbiory\DataSet_for_ANN-checkpoint.csv",delimiter=",")
bank_full = pd.read_csv(r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\bank-full.csv", delimiter=";")
bank_full["job"] = np.where(bank_full["job"]=="unknown","",bank_full["job"])
print(bank_full.isnull().sum())

from Fill_Missing import fill,count
# data = fill(bank_main)
# data2  = fill(bank_full)



#### dane sprawdzane czy nie trzeba wyrzucić któejś kolumny

print(heart.shape)
data =count(heart)
print(data.shape)
