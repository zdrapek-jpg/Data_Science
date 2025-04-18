import numpy as np
import pandas as pd


class OneHotEncoder:
    __slots__ = ["decoded_set","label_code","number_of_coded_keys"]
    def __init__(self,data):
        self.decoded_set = []
        self.label_code={}
        self.number_of_coded_keys=0
        pass

    def code_keys(self,data):
        """
            :parameter data pd.DataFrame
            :return dict like :{"a": [1,0,0], "b":[0,1,0] .....}
            """
        set_label = list(set(data))  # {a,b,c,d,e}
        #  z {a,b,c,d,e} przechodzimy na => {1,2,3,4,5}
        self.label_code= {i:el for i,el in enumerate(set_label)}

        self.decoded_set = {el:[]  for el in set_label } #{a:[],b:[],c:[],d:[],e:[]}
        self.number_of_coded_keys = self.decoded_set.__len__()
        numpy_zeros_like_decode_set = np.zeros([len(set_label),len(set_label)]) # macierz w któej definiujemy jedynki i przypisujemy do setu
        i = 0

        for row,key in zip(numpy_zeros_like_decode_set[:],self.decoded_set.keys()):
            row[i] = 1
            # numpy array
            #self.decoded_set[key] = row
            #lista nie array
            self.decoded_set[key]=row.tolist()
            i +=1
        return self.decoded_set
    #zamiast data podać rozmiar pierwszego wymmiaru danych

    def code_y_for_network(self, data):
        """
        :argument data in pd.DataFrame to transform
        :return data frame with keys() from code_keys and data splited with sepecific labels
        """
        new_data_Frame_coded = pd.DataFrame(columns=self.decoded_set.keys())
        #print(new_data_Frame_coded)
        for i in range(len(data)):
            new_data_Frame_coded.loc[i] = self.decoded_set[data[i]]

        #print(new_data_Frame_coded)
        #zwraca dataframe który ma rozmiar  ilość wierszy X ilość klas
        return new_data_Frame_coded


    def decode_keys(self,y_pred):
        """
          :param: y pred is list - sequence of soft max predictions of classes
          max argument (1) and return dict  :{"a": [1,0,0], "b":[0,1,0] .....}
             :return for [0,1,0,0] ->b
             """
        index_max =np.argmax(y_pred)
        #print(index_max)
        if 0 <= index_max <1:  # Avoid IndexError
           return self.label_code[index_max]
        else: return self.label_code[index_max]
        raise Exception
        # {"a": [1,0,0], "b":[0,1,0] .....} odkodowanie etykiet klas  jak model przewiduje do porónania

