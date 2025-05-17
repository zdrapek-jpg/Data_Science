#! Pyton 10.0
# load_user_data.py
from Data.One_hot_Encoder import OneHotEncoder
import numpy as np
import pandas as pd
import sys
import os
from Data.Decorator_time_logging import log_execution_time


@log_execution_time
def modify_user_input_for_network(data,one_hot_instance=None,standarization_instance=None, klucze=['Surname', 'CreditScore',
                                          'Country', 'Gender', 'Tenure', 'HasCrCard',
                                          'IsActiveMember', 'EstimatedSalary', 'age', 'job',
                                          'marital','education', 'balance', 'loan',],):
## ['CreditScore', 'Country', 'Gender', 'Tenure', 'HasCrCard','IsActiveMember',
# 'EstimatedSalary', 'age', 'job', 'marital','education', 'balance', 'loan']
    merged_data = pd.DataFrame(
        [data],
        columns=klucze
    )
    merged_data["loan"] = np.where(merged_data["loan"] == "yes", 1, 0)


    merged_data["HasCrCard"] = pd.to_numeric(merged_data.loc[:, "HasCrCard"])
    merged_data["Gender"] = np.where(merged_data["Gender"] == "Male", 1, 0)
    col1one_hot = [2, 9, 10, 11]
    # kolumna za pomocą normalizacji danych
    col2_norm = [1, 4, 6, 7, 8, 12]
    # kolumna index 2,12 jest pod numeric
    # kolumna za pomocą normalizacji danych

    for_normalization = merged_data.iloc[:,col2_norm]
    for_one_hot = merged_data.iloc[:,col1one_hot]



    normalized = standarization_instance.standarization_one_point(for_normalization.values.tolist()[0])
    normalized = pd.DataFrame([normalized],columns=merged_data.columns[col2_norm].tolist())
    #print(for_normalization_input_obj)


    data_as_1_row = pd.DataFrame()

one_hot_instance.load_data()
    data_one_hot = one_hot_instance.code_y_for_network(for_one_hot)

    data_as_1_row=data_one_hot


    data_as_1_row["HasCrCard"] = merged_data["HasCrCard"].values
    data_as_1_row["loan"] = merged_data["loan"].values
    data_as_1_row["Gender"] = merged_data["Gender"].values
    data_as_1_row = pd.concat((data_as_1_row, normalized),axis=1)

    print(data_as_1_row)
    return data_as_1_row.values
#
# from json import loads, dumps
# from flask import Flask, request, jsonify
# from NeuralNetwork.Network_single_class import NNetwork
#
# user_data = ["Henryk", 619, "France", "Female", 2, 1, 1, 101348.88, 58, "management", "married", "tertiary", 6429, "no"]
# ready_for_model = modify_user_input_for_network(user_data)
# print(ready_for_model[0])
# model_instance = NNetwork.create_instance()
# prediction = model_instance.pred(ready_for_model[0])
# print(prediction)
#
# app = Flask(__name__)
# # Post
# @log_execution_time
# @app.route('/')
# def predict():
#     try:
#         #user_data = request.get_json()
#         #user_data = user_data["user_data"]
#
#         user_data = ["Henryk", 619, "France", "Female", 2, 1, 1, 101348.88, 58, "management", "married", "tertiary",
#                      6429, "no"]
#
#         if not user_data:
#             return "Data is empty"
#         if len(user_data) != 14:
#             return f"Data is different length than expected {len(user_data)} != 14"
#         user_data = ["Henryk", 619, "France", "Female", 2, 1, 1, 101348.88, 58, "management", "married", "tertiary", 6429, "no"]
#         ready_for_model = modify_user_input_for_network(user_data)
#
#         model_instance = NNetwork.create_instance()
#         prediction = model_instance.pred(ready_for_model[0])
#         print(prediction)
#         return str(prediction[0])
#     except Exception as e:
#         return f"<h1>Error: {str(e)}</h1>", 500
# #
# #
# if __name__ == "__main__":
#     app.run(debug=True)
#
