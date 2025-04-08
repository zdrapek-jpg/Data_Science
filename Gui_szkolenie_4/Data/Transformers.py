import numpy as np
import pandas as pd
from math import sqrt
from enum import Enum
from typing import Union


class StandardizationType(Enum):
    MEAN_SCORE = "mean_score"
    Z_SCORE = "standaryzacja_z_score"
    LOG_SCALING = "log_scalling"
    NORMALIZATION = "normalization"
"""
A class for performing different types of data transformations.
"""
class Transformations:


    __slots__ = ["data","minimums","maximums","srednie","odchylenia_w_kolumnach","std_type"]
    def __init__(self,data=None,std_type:Union[StandardizationType,str]=StandardizationType.NORMALIZATION):
        """
                Parameters:
                - data : The dataset to transform.
                - std_type (StandardizationType): The type of standardization. Options:
                    - StandardizationType.MEAN_SCORE
                    - StandardizationType.Z_SCORE
                    - StandardizationType.LOG_SCALING
                    - StandardizationType.NORMALIZATION (default)
                """
        self.data = data
        self.minimums= []
        self.maximums = []
        self.srednie = []
        self.odchylenia_w_kolumnach = []
        self.std_type=std_type


    def srednia(self,sequence_like_column):
        s = 0
        # jeśli przekazany obiekt jest Listą , obiektem np lub pd  iterujemy się po nim
        if isinstance(sequence_like_column, (list ,np.ndarray,pd.core.frame.DataFrame,pd.Series)):

            for el in sequence_like_column:
                s += el
            # print(round(s,6))
            return 0 if -0.00001 < round(s / len(sequence_like_column), 5) < 0.00001 else round(
                s / len(sequence_like_column), 5)
        # jeśli obiekt nie jest obiektem tylko pojedyńczą wartością
        # jeśli wartość istnieje zwracamy ją w przeciwnym razie zwracamy 0
        return sequence_like_column if sequence_like_column >0 else 0
    def odchylenie_standardowe(self,sr,dane):
        suma= 0
        for i in dane:
           suma+= (i-sr)**2
        std =sqrt(suma/len(dane))
        self.odchylenia_w_kolumnach.append(std)
        return std

    def standarization_of_data(self,data):
        self.minimums = data.min()
        self.maximums = data.max()
        new_data = pd.DataFrame(columns= data.keys())
        klucze = list(data.keys())

        if self.std_type.value=="normalization":

            for i, (MIN,MAX) in enumerate(zip(self.minimums,self.maximums)):
                list_containing_column_values = data.iloc[:,i].values.tolist()
                new_data[klucze[i]]= [ round((x- MIN)/(MAX-MIN),6) for x in list_containing_column_values]

        elif self.std_type.value=="mean_score":
            for i, (MIN,MAX) in enumerate(zip(self.minimums,self.maximums)):
                list_containing_column_values = data.iloc[:,i].values.tolist()
                srednia_kolumny= self.srednia(list_containing_column_values)
                self.srednie.append(srednia_kolumny)
                new_data[klucze[i]]= [(x- srednia_kolumny)/(MAX-MIN) for x in list_containing_column_values]

        elif self.std_type.value =="standaryzacja_z_score":
            for i in range(data.shape[-1]):
                list_containing_column_values = data.iloc[:, i].values.tolist()
                srednia_kolumny = self.srednia(list_containing_column_values)
                std =self.odchylenie_standardowe(srednia_kolumny,list_containing_column_values)
                new_data[klucze[i]] = [(x -srednia_kolumny)/std for x in list_containing_column_values]

        elif self.std_type.value =="log_scalling" :
            granica_0 =np.finfo(float).eps()
            for i in range(data.shape[-1]):
                list_containing_column_values = data.iloc[:, i].values.tolist()
                new_data[klucze[i]]= [np.log(x+granica_0) for x in list_containing_column_values]
        return new_data


    # punkt do przetworzenia
    def normalize_one_point(self, point):
        if not (isinstance(point, list)):
            raise "element podany do zbioru musi być listą "
        if len(point) != len(self.minimums):
            raise f" ilosc elementów w zbiorze :{len(self.minimums)} != point( {len(point)})"

        if self.std_type.lower == "normalization":

            pass
        elif self.std_type.lower == "stadaryzacja_min_max":
            pass
        elif self.std_type.lower == "standaryzacja_z_score":
            pass
        elif self.std_type.lower == "log_scalling":
            pass
        else:
            pass
        #return [point[i] - self.minimums[i] / (self.maximums[i] - self.minimums[i]) for i in range(len(self.minimums))]






