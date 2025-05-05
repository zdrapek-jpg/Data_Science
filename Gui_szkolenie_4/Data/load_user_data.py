#! Pyton 10.0
# load_user_data.py
from One_hot_Encoder import OneHotEncoder
import numpy as np
import pandas as pd
import sys
from NeuralNetwork.Network_single_class import NNetwork
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



def modify_user_input_for_network(data, klucze=['Surname', 'CreditScore',
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
    for_one_hot = merged_data.iloc[:,col1one_hot].values.tolist()


    from Transformers import StandardizationType, Transformations
    # std zawiera informacje które są wykorzystywane przy transformacji punktu
    std = Transformations.load_data()
    #print(type(std),print(std.srednie))

    for_normalization_input_obj = std.standarization_one_point(for_normalization.values.tolist()[0])
    for_normalization_input_obj = pd.DataFrame([for_normalization_input_obj],columns=merged_data.columns[col2_norm].tolist())
    #print(for_normalization_input_obj)


    data_as_1_row = pd.DataFrame()

    one_hot = OneHotEncoder(merged_data)
    list_of_codes = one_hot.load_data()
    for dane_pod_kolumne, data_for_transform, klucz_do_nazwy_wartosci in zip(list_of_codes,for_one_hot[0],merged_data.columns[col1one_hot].tolist() ):
        #print(dane_pod_kolumne,data_for_transform,klucz_do_nazwy_wartosci)
        #dane_pod_kolumne = Klucz_do_nazwy_wartosci: {klucz1 : [wartosci row], klucz2 : [0,1,0,0]}, klucz_do_nazwy_wartosci}
        one_hot = OneHotEncoder(merged_data)
        one_hot.decoded_set = dane_pod_kolumne[klucz_do_nazwy_wartosci]

        sett = one_hot.code_y_for_network(data_for_transform)
        data_as_1_row = pd.concat((data_as_1_row, sett), axis=1)
    #print(sett)

    data_as_1_row["HasCrCard"] = merged_data["HasCrCard"].values
    data_as_1_row["loan"] = merged_data["loan"].values
    data_as_1_row["Gender"] = merged_data["Gender"].values
    data_as_1_row = pd.concat((data_as_1_row, for_normalization_input_obj),axis=1)

    #print(data_as_1_row)
    return data_as_1_row.values

from json import loads, dumps
from flask import Flask, request, jsonify
from NeuralNetwork.Network_single_class import NNetwork


app = Flask(__name__)

# Post
@app.route('/')
def predict():
    user_data = ["Henryk", 619, "France", "Female", 2, 1, 1, 101348.88, 58, "management", "married", "tertiary", 6429,
                 "no"]
    ready_for_model = modify_user_input_for_network(user_data)
    model_instance = NNetwork.create_instance()
    prediction = model_instance.pred(ready_for_model)
    return prediction[0]



if __name__ == "__main__":
    app.run(debug=True)

