import numpy as np
import pandas as pd

from Data.One_hot_Encoder import OneHotEncoder

def load_data(file_path=r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\new_data.csv"):
    merged_data = pd.read_csv(file_path,
                              delimiter=";")
    return merged_data

def process_standarization(data_containing_required_columns):
    from Data.Transformers import StandardizationType, Transformations
    # std zawiera informacje które są wykorzystywane przy transformacji punktu
    std = Transformations(data_containing_required_columns, StandardizationType.Z_SCORE)
    trained_data = std.standarization_of_data(data_containing_required_columns)
    ## save data
    std.save_data() # file_path we can define path to set

    new_std = Transformations.load_data() #  we need to give path we used to save_data
    return trained_data
def process_one_hot_encoder(data_containing_required_columns):
    pass

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
    normalized = process_standarization(for_normalization)
    for_one_hot = merged_data.iloc[:, col1one_hot]


    data_set = pd.DataFrame()
    code = []
    for k in for_one_hot.keys():
        all_decoded_sets ={}
        data = for_one_hot.loc[:, k].values

        one_hot = OneHotEncoder(data)
        one_hot.code_keys(data)
        all_decoded_sets[k] =one_hot.decoded_set
        all_decoded_sets["label_code"]= one_hot.label_code
        all_decoded_sets["number_of_coded_keys"] = one_hot.number_of_coded_keys
        code .append(all_decoded_sets)




        ### potrzebuje tylko decoded set ponieważ on zaweira dane w
        ## 'unknown': [1.0, 0.0, 0.0, 0.0], 'secondary': [0.0, 1.0, 0.0, 0.0] ...
        #print(one_hot.decoded_set)
        data_modified = one_hot.code_y_for_network(data)
        #print(data_modified)
        data_set = pd.concat((data_set, data_modified), axis=1)
    one_hot.save_data(code)
    data_set["HasCrCard"] = merged_data["HasCrCard"].values
    data_set["loan"] = merged_data["loan"].values
    data_set["Gender"] = merged_data["Gender"].values
    #print(code,sep="\n")


    data_set = pd.concat((data_set, normalized), axis=1)
    y.replace(("yes", "no"), (1, 0), inplace=True)
    data_set["y"] = y
    return data_set

#
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
