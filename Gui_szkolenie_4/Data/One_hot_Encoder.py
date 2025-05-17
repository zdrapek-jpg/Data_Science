import numpy as np
import pandas as pd
import json

class OneHotEncoder:
    __slots__ = ["decoded_set","label_code","number_of_coded_keys","data_code"]
    def __init__(self):
        self.decoded_set = {}
        self.label_code={}
        self.number_of_coded_keys=0
        self.data_code={}


    def code_keys(self,data):
        """
               :argument data in pd.DataFrame (one column)! to transform
               :return data frame with keys() from code_keys and data splited with sepecific labels
        """
        if isinstance(data, pd.Series):
            data = data.to_frame()
        for column_name in data.columns:
            unique_elements =list(set(data.loc[:,column_name].values.tolist()))

            self.label_code = {i: unique_element for i, unique_element in enumerate(unique_elements)}


            self.decoded_set ={unique_element:None   for unique_element in self.label_code.values()}
            self.number_of_coded_keys = self.decoded_set.__len__()
            numpy_zeros_shape_like_num_of_coded_keys=np.zeros((self.number_of_coded_keys,self.number_of_coded_keys))
            for i,(row,unique_value) in enumerate(zip(numpy_zeros_shape_like_num_of_coded_keys[:],self.decoded_set.keys())):
                row[i]=1
                self.decoded_set[unique_value]=list(row)
            self.data_code[column_name]={"label_code":self.label_code,
                                      "decoded_set":self.decoded_set,
                                      "number_of_coded_keys":self.number_of_coded_keys
                                      }

    def label_encoder_keys(self,data,orders:list):
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

    def code_y_for_network(self,data):
        """
                  :param: data is pd.DataFrame or pd.Series for coding in ohe hot
                  max argument (1) and return dict  :{"a": [1,0,0], "b":[0,1,0] .....}
                     :return for [0,1,0,0] ->b in pd.DataFrame
                     """
        new_data_frame_data= pd.DataFrame()
        for column_name in data.columns:


            single_data_for_code = data[column_name]

            actual_data_for_coding = self.data_code[column_name]
            #print(list(actual_data_for_coding["decoded_set"].values())[0])
            if isinstance(list(actual_data_for_coding["decoded_set"].values())[0],(int,float)):
                new_data_frame_colums = pd.DataFrame(columns=[column_name])
            else:
                #print(list(actual_data_for_coding["decoded_set"].keys()))
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
    def save_data(self,path =r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\test_one_hot.json"):

        with open(path,"w") as file_write:
            json.dump(self.data_code,file_write,indent=2)
    @classmethod
    def load_data(cls,path=r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\test_one_hot.json"):
        try:
            with open(path,"r") as read_file:
                data = json.load(read_file)
                new_one_hot_instance = cls()
                new_one_hot_instance.data_code = data
                return new_one_hot_instance
        except FileNotFoundError:
            print(f"File not found: {path}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {path}")














