
import pandas as pd
class SplitData:
    train = 0.4
    valid = 0.4
    test = 0.2


    """
    :parameter train , valid,tests should sum to 1.0
    :param paramters defind how data will be splited with split_data()
    """
    @classmethod
    def set(cls,train=0.4,valid=0.4,test=0.2):
        cls.train = train  # 0.40
        cls.valid = valid  # 0.40
        cls.test = test  # 0.20



    @classmethod
    def split_data(cls,data):
        """data must be in data frame where [ x] :
        and last column is named label and it reprezented by y
        data can be normalized bofor spliling
        basic setting is train = 0.4 valid =0 .4 and test=0.2"""
        if cls.train+cls.valid+cls.test<1:
            cls.train+= 1 - (cls.valid+cls.test)
        if  0>cls.train+cls.valid+cls.test>1:
            raise "nie można podzielić zbioru podano podział który uwzględdnia ponad 100 % zbioru"
        shuffled_data = data.sample(frac =1).reset_index(drop=True)

        # podzielenie na x i y
        # domyślnie że y jest ostanią kolumną
        len_of_data = shuffled_data.shape[0]
        split1 = int(len_of_data*cls.train)
        split2 = int(len_of_data*cls.valid)+split1
        shuffled_data.rename(columns={shuffled_data.columns[-1]: "y"}, inplace=True)

        x_train,y_train = shuffled_data.iloc[:split1,:-1].values,       shuffled_data.iloc[:split1,-1].values
        x_valid,y_valid = shuffled_data.iloc[split1:split2,:-1].values, shuffled_data.iloc[split1:split2,-1].values
        x_test,y_test = shuffled_data.iloc[split2:,:-1].values,         shuffled_data.iloc[split2:,-1].values
        return x_train,y_train,x_valid,y_valid,x_test,y_test

    @classmethod
    def tasowanie(cls,data,f= False):
        """
        :param data is pd.DataFrame object
        :param f  when is set to False it splits data for x and y
        :param f  is set to True it returns pd.DataFrame
        """
        shuffled_data = data.sample(frac=1).reset_index(drop=True)
        if f:
            return shuffled_data
        return shuffled_data.iloc[:, :-1].values, shuffled_data.loc[:, "y"]

    @classmethod
    def merge(cls,x,y):
        """
        :return x and y merged into one dataframe with last colum named y"""

        data = pd.DataFrame(x, columns=[f"x{i}" for i in range(x.shape[1])])
        data["y"] = y
        return data

    @classmethod
    def batch_split_data(cls,data_frame,size):
        """
          :param dataframe is pd.DataFrame
          :param size defines size of every batch

          function shuffle data and cretates batches in list of x nad y
          :return list of batches  x,y
          """
        shuffled_data = SplitData.tasowanie(data_frame,f=True)

        # ile jest elementów
        len_data =shuffled_data.shape[0]
        # ile na 1 batch
        podzial = int(len_data // size)
        #tworzymy mini batche

        start, stop = 0, podzial
        batch_x, batch_y = [], []
        for i in range(size):
            batch_x.append(data_frame.iloc[start:stop, :-1].values)
            batch_y.append(data_frame.iloc[start:stop,-1].values)
            start += podzial
            stop += podzial
        batch_x.append(data_frame.iloc[start:,:-1].values)
        batch_y.append(data_frame.iloc[start:,-1].values)
        return batch_x,batch_y


