import numpy as np
import pandas as pd
from math import pow,sqrt


class Transformations:
    def __init__(self,data=None):
        self.data = data
        self.minimums= []
        self.maximums = []
        self.srednie = []
        self.odchylenia_w_kolumnach = []

    def srednia(self,sequence_like_colum):
        s = 0
        if  isinstance(sequence_like_colum,list) or isinstance(sequence_like_colum,np.ndarray) or isinstance(sequence_like_colum,pd.core.frame.DataFrame):

            for el in sequence_like_colum:
                s+=el
            #print(round(s,6))
            return 0 if -0.00001 < round(s / len(sequence_like_colum), 5) < 0.00001 else   round(s / len(sequence_like_colum), 5)
        return 0
    def odchylenie_standardowe(self,sr,dane):
        suma= 0
        for i in dane:
           suma+= (i-sr)**2
        std =sqrt(suma/len(dane))
        self.odchylenia_w_kolumnach.append(std)
        return std


    def standaryzacja_mean_score(self,data,y):
        self.minimums = data.min()
        self.maximums = data.max()
        new_data = pd.DataFrame()
        for i, (MIN, MAX) in enumerate(zip(self.minimums, self.maximums)):

            lista_el = data.iloc[:, i].values.tolist()
            sr =self.srednia(lista_el)
            self.srednie.append(sr)
            new_data[f"x{i}"] = [(x - sr) / (MAX - MIN) for x in lista_el]
        return new_data



    def standaryzacja_z_score(self,data,flag=True,odciecie=None):
        new_data = np.zeros((data.shape[0],data.shape[-1]-1,), dtype=float)
        for i in range(data.shape[-1]-1):
            sr =self.srednia(data[:,i])
            self.srednie.append(sr)
            std =self.odchylenie_standardowe(sr,data[:,i])
            new_data[:,i] =([(x -sr)/std for x in data[:,i]])
        new_data = np.column_stack((new_data,data[:,-1]))
        if odciecie is None and flag:
            return new_data

            # Filter rows where absolute values exceed the cutoff (excluding the last column)
        filtered_data = new_data[np.all(np.abs(new_data[:, :-1]) <= odciecie, axis=1)]

        return filtered_data



    def stadaryzacja_min_max_normalization(self,data):
        self.minimums=data.min()
        self.maximums= data.max()
        i = 0
        new_data =pd.DataFrame()
        for i,(MIN,MAX) in enumerate(zip(self.minimums,self.maximums)):
            lista_el = data.iloc[:,i].values.tolist()
            new_data[f"x{i}"]=[round((x -MIN)/(MAX-MIN),5) for x in lista_el]
        return new_data

    def normalize_one_point(self,point):
        if not (isinstance(point,list)):
            raise "element podany do zbioru musi być listą "
        if len(point)!= len(self.minimums):
            raise f" ilosc elementów w zbiorze :{len(self.minimums)} != point( {len(point)})"
        return [point[i]-self.minimums[i]/(self.maximums[i]-self.minimums[i]) for i in range(len(self.minimums))]


##### bardzo ważne do spłaszczania list przy deformacjach one hot encoder  itp
def flatten_list(lista):
    big = []
    if not isinstance(lista,list):
        return lista
    for el in lista:
        if  isinstance(el,list):
            big.extend(flatten_list(el))
        else:
            big.append(el)
    return big