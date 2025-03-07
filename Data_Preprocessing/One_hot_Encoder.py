import numpy as np
class OneHotEncoder:
    def __init__(self,data):
        self.decoded_set = []
        self.label_code={}
        pass

    """
    :return dict like :{"a": [1,0,0], "b":[0,1,0] .....}
    """
    def code_keys(self,data):
        set_label = list(set(data))  # {a,b,c,d,e}
        #  z {a,b,c,d,e} przechodzimy na => {1,2,3,4,5}
        self.label_code= {i:el for i,el in enumerate(set_label)}

        self.decoded_set = {el:[]  for el in set_label } #{a:[],b:[],c:[],d:[],e:[]}
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

    def code_y_for_network(self, data):
        for i in range(len(data)):
            data[i] = np.array([self.decoded_set[data[i]]])
        return data

    def decode_keys(self,y_pred):
        index_max =np.argmax(y_pred)
        #print(index_max)
        if 0 <= index_max <1:  # Avoid IndexError
           return self.label_code[index_max]
        else: return self.label_code[index_max-1]
        raise Exception
        # {"a": [1,0,0], "b":[0,1,0] .....} odkodowanie etykiet klas  jak model przewiduje do porónania

