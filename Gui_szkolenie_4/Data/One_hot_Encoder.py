import numpy as np
import pandas as pd
import json

class OneHotEncoder:
    __slots__ = ["decoded_set","label_code","number_of_coded_keys","data_code"]
    def __init__(self,data):
        self.decoded_set = {}
        self.label_code={}
        self.number_of_coded_keys=0
        self.data_code={}


        pass

    def code_keys(self,data):
        """
            :parameter data pd.DataFrame
            :return
            : decoded_set =  dict like :{"a": [1,0,0], "b":[0,1,0] .....}

            """

        #### dorobić możliwość iteracji po wszystkich kolumnach całego zbioru wrzucanego tak jak jest w load_data
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
        :argument data in pd.DataFrame (one column)! to transform
        :return data frame with keys() from code_keys and data splited with sepecific labels
        """
        new_data_Frame_coded = pd.DataFrame(columns=self.decoded_set.keys())
        #print(new_data_Frame_coded)
        if isinstance(data, str):
            new_data_Frame_coded.loc[0] = self.decoded_set[data]
            return new_data_Frame_coded
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


    def save_data(self,all_decoded_sets,path=r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\one_hot_encoder_data.json"):
        with open(path, "w") as write_file:
                json.dump(all_decoded_sets, write_file, indent=1)
    @classmethod
    def load_data(cls):
        file_name = r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\one_hot_encoder_data.json"
        try:
            with open(file_name, "r") as file_read:
                return json.load(file_read)  # data_list is a list of dicts!

        except FileNotFoundError:
            print(f"File not found: {file_name}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_name}")



    def new_one_hot_encoder_keys(self,data):
        if isinstance(data, pd.Series):
            data = data.to_frame()
        for column_name in data.columns:
            unique_elements =list(set(data.loc[:,column_name].values.tolist()))

            self.label_code = {i: unique_element for i, unique_element in enumerate(unique_elements)}


            self.decoded_set ={unique_element:None   for unique_element in self.label_code.values()}
            self.number_of_coded_keys = self.decoded_set.__len__()
            numpy_zeros_shape_like_num_of_coded_keys=np.zeros((self.number_of_coded_keys,self.number_of_coded_keys),dtype=int)
            for i,(row,unique_value) in enumerate(zip(numpy_zeros_shape_like_num_of_coded_keys[:],self.decoded_set.keys())):
                row[i]=1
                self.decoded_set[unique_value]=row
            self.data_code[column_name]={"label_code":self.label_code,
                                      "decoded_set":self.decoded_set,
                                      "number_of_coded_keys":self.number_of_coded_keys
                                      }
    def new_label_encoder_keys(self,data,orders:list):
        """
            :argument data in pd.DataFrame! to transform
            :argument orders list of list that contain order of label encoder
        """
        if isinstance(data, pd.Series):
            data = data.to_frame()
        for column_name,order_of_column in zip(data.columns,orders):


            self.label_code = {i: unique_element for i, unique_element in enumerate(order_of_column)}

            self.decoded_set = {unique_element:value  for value,unique_element in self.label_code.items()}
            self.number_of_coded_keys = self.decoded_set.__len__()
            self.data_code[column_name] = {"label_code": self.label_code,
                                           "decoded_set": self.decoded_set,
                                           "number_of_coded_keys": self.number_of_coded_keys
                                           }

    def new_code_y_for_network(self,data):
        new_data_frame_data= pd.DataFrame()
        for column_name in data.columns:


            single_data_for_code = data[column_name]

            actual_data_for_coding = self.data_code[column_name]
            print(list(actual_data_for_coding["decoded_set"].values())[0])
            if isinstance(list(actual_data_for_coding["decoded_set"].values())[0],(int,float)):
                new_data_frame_colums = pd.DataFrame(columns=[column_name])
            else:
                print(list(actual_data_for_coding["decoded_set"].keys()))
                new_data_frame_colums = pd.DataFrame(columns=list(actual_data_for_coding["decoded_set"].keys()))
            #print(new_data_frame_colums)


            i=0
            for value in single_data_for_code:
                #print(value,end= " ")
                new_data_frame_colums.loc[i]=actual_data_for_coding["decoded_set"][value]
                #print(actual_data_for_coding["decoded_set"][value])
                i+=1
            new_data_frame_data = pd.concat((new_data_frame_data,new_data_frame_colums),axis=1)
        return  new_data_frame_data










