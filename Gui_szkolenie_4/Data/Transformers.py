import numpy as np
import pandas as pd
from math import sqrt
from enum import Enum
from typing import Union
import json


class StandardizationType(Enum):
    """
    Attributes:
    - MEAN_SCORE: Standardization by centering data around the mean (mean centering).
    - Z_SCORE: Z-score standardization (subtract mean, divide by standard deviation).
    - LOG_SCALING: Logarithmic scaling to reduce skewness or compress large values.
    - NORMALIZATION: Rescales data to a fixed range, typically [0, 1].
    """
    MEAN_SCORE = "mean_score"
    Z_SCORE = "standaryzacja_z_score"
    LOG_SCALING = "log_scalling"
    NORMALIZATION = "normalization"

class Transformations:
    """
    A class for performing different types of data transformations.
    """


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
        """

        :param sequence_like_column:
        :return: mean of sequence or point or return 0
        """
        s = 0
        # jeśli przekazany obiekt jest Listą , obiektem np lub pd  iterujemy się po nim
        if isinstance(sequence_like_column, (list ,np.ndarray,pd.core.frame.DataFrame,pd.Series)):
            s=0
            for el in sequence_like_column:
                s += el
            # print(round(s,6))
            return 0 if -0.00001 < round(s / len(sequence_like_column), 5) < 0.00001 else round(
                s / len(sequence_like_column), 5)

        # jeśli obiekt nie jest obiektem tylko pojedyńczą wartością
        # jeśli wartość istnieje zwracamy ją w przeciwnym razie zwracamy 0
        return sequence_like_column if sequence_like_column >0 else 0
    def odchylenie_standardowe(self,sr,dane):
        """

        :param sr:  mean of actual column
        :param dane:  kolumna po której liczymy
        :return:  odchylenie standardowe danych
        """
        suma= 0
        for i in dane:
           suma+= (i-sr)**2
        std =sqrt(suma/len(dane))
        self.odchylenia_w_kolumnach.append(std)
        return std

    def standarization_of_data(self,data):
        """

        :param data:  sequence in pd.DataFrame
        :return:  data standarized
        """
        self.minimums = data.min().values.tolist()
        self.maximums = data.max().values.tolist()
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
        """

        :param point: apply normalization with saved data for normalization
        :return:
        """
        if not (isinstance(point, list)):
            raise "element podany do zbioru musi być listą "



        if self.std_type.lower == "normalization":
            assert len(point) == len(self.minimums) == len(self.maximums),f"data {len(point)} == {len(self.minimums)} == {len(self.maximums)} not equal"
            return [round((x - MIN) / (MAX - MIN), 6) for x,MIN,MAX in zip(point,self.minimums,self.maximums)]

        elif self.std_type.lower == "mean_score":
            assert len(point) == len(self.minimums) == len(
                self.maximums)==len(self.srednie), f"data {len(point)} == {len(self.minimums)} == {len(self.maximums)} == {len(self.srednie)} not equal"
            return  [(x - srednia_kolumny) / (MAX - MIN) for x,srednia_kolumny,MIN,MAX in zip(point,self.srednie,self.minimums,self.maximums)]

        elif self.std_type.lower == "standaryzacja_z_score":
            assert len(point) == len(self.srednie) == len(
                self.odchylenia_w_kolumnach), f"data {len(point)} == {len(self.srednie)} == {len(self.odchylenia_w_kolumnach)} not equal"
            return  [(x - srednia_kolumny) / std for x,srednia_kolumny,std in zip(point,self.srednie,self.odchylenia_w_kolumnach)]

        elif self.std_type.lower == "log_scalling":
            granica_0 = np.finfo(float).eps()
            return [np.log(x + granica_0) for x in point]

        else:
            return Exception

    import json

    def save_data(self,
                  file_name=r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\transformers_data.json"):
        # Create a dictionary with named keys
        data = {
            "minimums": self.minimums,
            "maximums": self.maximums,
            "srednie": self.srednie,
            "odchylenia_w_kolumnach": self.odchylenia_w_kolumnach
        }

        # Filter out keys where the value is None or empty
        filtered_data = {k: v for k, v in data.items() if  len(v) >= 1}

        # Only add std_type if there's other data
        filtered_data["std_type"] = self.std_type.value
        print(filtered_data)
        with open(file_name,"w")as file_write:
            json.dump(filtered_data,file_write,indent=5)














