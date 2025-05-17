import numpy as np
import pandas as pd
import threading
from Data.Decorator_time_logging import *
from Data.One_hot_Encoder import OneHotEncoder

def load_data(file_path=r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\new_data.csv"):
    merged_data = pd.read_csv(file_path,
                              delimiter=";")
    return merged_data
@log_execution_time
def process_standarization(data_containing_required_columns,result_dict,key):
    from Data.Transformers import StandardizationType, Transformations
    # std zawiera informacje które są wykorzystywane przy transformacji punktu
    std = Transformations(data_containing_required_columns, StandardizationType.Z_SCORE)
    trained_data = std.standarization_of_data(data_containing_required_columns)
    ## save data
    std.save_data() # file_path we can define path to set
    result_dict[key] =trained_data
    return trained_data
@log_execution_time
def process_one_hot_encoder(data_containing_required_columns, result_dict, key):
    one_hot = OneHotEncoder()
    one_hot.code_keys(data_containing_required_columns)
    data_one_hot = one_hot.code_y_for_network(data_containing_required_columns)
    one_hot.save_data()
    result_dict[key] = data_one_hot
def data_preprocessing():
    merged_data = load_data()

    merged_data["HasCrCard"] = pd.to_numeric(merged_data.loc[:, "HasCrCard"])
    merged_data["loan"] = np.where(merged_data["loan"] == "yes", 1, 0)
    merged_data["Gender"] = np.where(merged_data["Gender"] == "Male", 1, 0)
    #print(pd.unique(merged_data["HasCrCard"]))


    # kolumna za pomocą normalizacji danych
    col2_norm = [1, 4, 6, 7, 8, 12]
    y = merged_data.loc[:, "y"]
    ## dane dla one hot encodera
    col1one_hot = [2, 9, 10, 11]
    # 13 i 14 jest poprostu zamieniana  na 0 lub 1 więc nie trzeba używać one hot encodera

    for_normalization = merged_data.iloc[:, col2_norm]
    for_one_hot = merged_data.iloc[:, col1one_hot]

    result_dict = {}
    thread1 = threading.Thread(target=process_standarization, args=(for_normalization, result_dict, "normalized"))
    thread2 = threading.Thread(target=process_one_hot_encoder, args=(for_one_hot, result_dict, "one_hot"))


    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    # Retrieve results
    normalized = result_dict.get("normalized")
    data_one_hot = result_dict.get("one_hot")

    if normalized is None or data_one_hot is None:
        logging.error(f"Data processing failed. norm: {type(normalized)},one hot: {type(data_one_hot)} Exiting...")
        return None

    data_set = pd.DataFrame()
    data_set=pd.concat((data_one_hot,data_set),axis=1)
    data_set["HasCrCard"] = merged_data["HasCrCard"].values
    data_set["loan"] = merged_data["loan"].values
    data_set["Gender"] = merged_data["Gender"].values
    #print(code,sep="\n")

    data_set = pd.concat((data_set, normalized), axis=1)
    y.replace(("yes", "no"), (1, 0), inplace=True)
    data_set["y"] = y
    #print(data_set.shape)
    logging.info(f"Data preprocessing completed. Dataset shape: {data_set.shape}")
    return data_set


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
