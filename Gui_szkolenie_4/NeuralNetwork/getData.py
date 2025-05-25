import numpy as np
import pandas as pd
import threading
from Data.Decorator_time_logging import *
from Data.One_hot_Encoder import OneHotEncoder
from Data.Transformers import StandardizationType

def load_data(file_path=r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\new_data.csv"):
    merged_data = pd.read_csv(file_path,
                              delimiter=";")
    return merged_data
@log_execution_time
def process_standarization(data_containing_required_columns,type:StandardizationType,save_model=True):
    """
    :param data_containing_required_columns pd.DataFrame that contains required columns for standarization:
    :param type:enum StandardizationType enum that defines type of standarization algorithm
    :param save_model:  if true model parameters are saved into a file
    :return:  pd.DataFrame with same column names and values standarized
    If save model is False function try to load data from file and create instance of class Transformation
    """
    from Data.Transformers import  Transformations
    if not save_model:
        std_model = Transformations.load_data()
        normalized_data =pd.DataFrame(columns=data_containing_required_columns.columns)
        if data_containing_required_columns.shape[0]<=1:
           return  std_model.standarization_one_point(data_containing_required_columns.values.tolist())

        for idx, row in data_containing_required_columns.iterrows():
            standardized_values = std_model.standarization_one_point(row.values.tolist())
            normalized_data.loc[idx] = standardized_values
        return normalized_data
    else:
        # std zawiera informacje które są wykorzystywane przy transformacji punktu
        std = Transformations(data_containing_required_columns, type)
        normalizad_data = std.standarization_of_data(data_containing_required_columns)
    ## save data
    if save_model:
        std.save_data() # file_path we can define path to set
    return normalizad_data

@log_execution_time
def process_one_hot_encoder(data_containing_required_columns,data_containing_label_columns=None,save_model=True):
    """

    :param data_containing_required_columns:  pd.DataFrame containing data to process
    :param data_containing_label_columns:  pd.DataFrame containing columns for label encoder
    if label encoder is not [] we give list of object and socond list of orders
    :param save_model:  if true we create new instance on data passed else if param is False we load model form file and process one-hot-encoder
    :return: pd.DataFrame with one_hot encoder data and saved model
    [0,1,2,3,4,5,6,7,8,9,10],[0,1,2,3,4,5,6,7,8,9,10]
    """
    if not save_model:
        one_hot =OneHotEncoder.load_data()
    else:
        one_hot = OneHotEncoder()
        one_hot.code_keys(data_containing_required_columns)

    data_one_hot = one_hot.code_y_for_network(data_containing_required_columns)

    if data_containing_label_columns is not None:
        one_hot.label_encoder_keys(data_containing_label_columns,[[0,1,2,3,4,5,6,7,8,9,10],[0,1,2,3,4,5,6,7,8,9,10]])
        data_one_hot =pd.concat((data_one_hot,one_hot.code_y_for_network(data_containing_label_columns)),axis=1)

    if save_model:
        one_hot.save_data()

    return  data_one_hot


def data_preprocessing(std_column:list ,one_hot_column:list,label_encoder_column:list,std_type:StandardizationType,save_models=True,merged_data=None):
    if save_models:
        merged_data = load_data()

        merged_data["HasCrCard"] = pd.to_numeric(merged_data.loc[:, "HasCrCard"])
    merged_data["loan"] = np.where(merged_data["loan"] == "yes", 1, 0)
    merged_data["Gender"] = np.where(merged_data["Gender"] == "Male", 1, 0)
    merged_data["job"]=np.where(merged_data["job"]=="unknown","unknown1",merged_data["job"])
    # 13 i 14 jest poprostu zamieniana  na 0 lub 1 więc nie trzeba używać one hot encodera
    if std_column:
        for_normalization = merged_data.iloc[:, std_column]
        normalized = process_standarization(for_normalization, std_type, save_models)
    if one_hot_column:
        if label_encoder_column is not None:
            label_encoder_column = merged_data.iloc[:, label_encoder_column]
        for_one_hot = merged_data.iloc[:, one_hot_column]
        data_one_hot = process_one_hot_encoder(for_one_hot, label_encoder_column, save_models)




    print(normalized.shape)
    print(data_one_hot.shape)

    if normalized is None or data_one_hot is None:
        logging.error(f"Data processing failed. norm: {type(normalized)},one hot: {type(data_one_hot)} Exiting...")
        return None

    data_set = pd.DataFrame()
    data_set=pd.concat((data_one_hot,data_set),axis=1)
    data_set["HasCrCard"] = merged_data["HasCrCard"].values
    data_set["loan"] = merged_data["loan"].values
    data_set["Gender"] = merged_data["Gender"].values

    data_set = pd.concat((data_set, normalized), axis=1)
    print("size ",data_set.columns)
    logging.info(f"Data preprocessing completed. Dataset shape: {data_set.shape}")
    if save_models:
        data_set["y"]= np.where(merged_data["y"]=="yes",1,0)
        data_set.to_csv(path_or_buf=r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\training_data.csv",sep=";",columns=data_set.columns,index=False)
    return data_set
#
# print(data_preprocessing(std_column=[1, 6, 7, 8, 12,4],one_hot_column=[2, 9, 10, 11],label_encoder_column=[],std_type= StandardizationType.NORMALIZATION,save_models= True))







# import numpy as np
#
# # Inputs and labels
# X = np.array([[0, 0.5, 1],[0.5, 1.5, 1],[0.5, 1, 1.5],[0, 0.5, 1],[0.5, 1.5, 1],[0.5, 1, 0]])  # shape (2, 3)
# Y = np.array([1,0,1,1,0,0])         # shape (2,)
#
# # Initialize parameters
# wagi = np.random.rand(3)  *0.7     # 3 input features
# biasy = 0.0
# print(wagi)
# # Momentum terms
# Vderivative_like_weights = np.zeros_like(wagi)
# Vb_like_bias = 0.0
#
# # Hyperparameters
# alpha = 0.02    # learning rate
# beta_momentum = 0.9     # momentum
# epochs = 100
#
# # ReLU activation and its derivative
# def relu(z):
#     return np.maximum(0, z)
#
# def relu_derivative(z):
#     return (z > 0).astype(float)
#
#
# # Training loop
# for epoch in range(epochs):
#     dw = np.zeros_like(wagi)
#     db = 0.0
#     loss = 0
#
#     for i in range(len(X)):
#         point = X[i]
#         y = Y[i]
#
#         # Forward pass
#         product = np.dot(wagi, point) + biasy
#         output = relu(product)
#
#         # Compute loss (MSE)
#         loss += 0.5 * (output - y) ** 2
#
#         # Backward pass
#         pochodna_wyjścia = (output - y)
#         pochodna_aktywacji = relu_derivative(product)
#         gradient = pochodna_wyjścia*pochodna_aktywacji
#         dw += gradient * point
#         db += gradient
#
#     # Average gradients
#     dw /= len(X)
#     db /= len(X)
#     loss /= len(X)
#
#     # Momentum update
#     Vderivative_like_weights = beta_momentum * Vderivative_like_weights + (1- beta_momentum) * dw
#     Vb_like_bias = beta_momentum * Vb_like_bias + (1-beta_momentum) * db
#
#     wagi -= alpha * Vderivative_like_weights
#     biasy -= alpha * Vb_like_bias
#
#     if epoch % 10 == 0:
#         print(f"Epoch {epoch}, Loss: {loss:.4f}")
#
# for i in range(len(X)):
#     point = X[i]
#     y = Y[i]
#
#     # Forward pass
#     product = np.dot(wagi, point) + biasy
#     output = relu(product)
#
# print("\nFinal weights:", wagi)
# print("Final bias:", biasy)
