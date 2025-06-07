#! Pyton 10.0
# load_user_data.py
from Data.One_hot_Encoder import OneHotEncoder
import numpy as np
import pandas as pd

from Data.Transformers import Transformations

normalization_instance =None
one_hot_instance  = None
model_instance  = None

def load():
    global normalization_instance, one_hot_instance, model_instance
    normalization_instance = Transformations.load_data()  # instancja klasy normalizującej - ładujemy ją
    one_hot_instance = OneHotEncoder.load_data()  # plik json jest przypisane do klasy - tworzymy instancje klasy
    model_instance = NNetwork.create_instance()  # Create the model instance
    print(normalization_instance.std_type)
    print("Data loaded successfully!")

#@log_execution_time
def modify_user_input_for_network(data,one_hot_instance=None,standarization_instance=None, klucze=['Surname', 'CreditScore',
                                          'Country', 'Gender', 'Tenure', 'HasCrCard',
                                          'IsActiveMember', 'EstimatedSalary', 'age', 'job',
                                          'marital','education', 'balance', 'loan',],col2_norm=None,col1one_hot=None):
    merged_data = pd.DataFrame(
        [data],
        columns=klucze
    )
    merged_data["loan"] = np.where(merged_data["loan"] == "yes", 1, 0)
    merged_data["HasCrCard"] = pd.to_numeric(merged_data.loc[:, "HasCrCard"])
    merged_data["Gender"] = np.where(merged_data["Gender"] == "Male", 1, 0)

    col1one_hot = [2, 9, 10, 11]
    # kolumna za pomocą normalizacji danych
    col2_norm = [1, 6, 7, 8, 12,4]
    # kolumna index 2,12 jest pod numeric
    # kolumna za pomocą normalizacji danych
    for_normalization = merged_data.iloc[:,col2_norm]
    for_one_hot = merged_data.iloc[:,col1one_hot]
    normalized = standarization_instance.standarization_one_point(for_normalization.values.tolist()[0])

    normalized = pd.DataFrame([normalized],columns=merged_data.columns[col2_norm].tolist())
    data_as_1_row = pd.DataFrame()
    data_one_hot = one_hot_instance.code_y_for_network(for_one_hot)
    data_as_1_row=data_one_hot
    data_as_1_row["HasCrCard"] = merged_data["HasCrCard"].values
    data_as_1_row["loan"] = merged_data["loan"].values
    data_as_1_row["Gender"] = merged_data["Gender"].values
    data_as_1_row = pd.concat((data_as_1_row, normalized),axis=1)
    data_as_1_row
    # for k in data_as_1_row.keys():
    #     print(k,end=" ")
    # print()
    # for valu in data_as_1_row.values.tolist():
    #     print(valu,end=" ")
    # print()
    return data_as_1_row.values


from json import loads, dumps
from flask import Flask, request, jsonify
from NeuralNetwork.Network_single_class1 import NNetwork
#
# load()
# user_data = ["Boni", 699, "France", "Female", 1, 0, 0, 93826.63, 47, "blue-collar", "married", "unknown", 4518, "no"]
#
# ready_for_model = modify_user_input_for_network(user_data,one_hot_instance,normalization_instance)
# print(ready_for_model[0])
# prediction = model_instance.pred(ready_for_model[0])
# data = pd.read_csv(r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\training_data.csv",delimiter=";")
# x= data.iloc[:,:-1].values
# y = data.iloc[:,-1].values
#
# acc,error = model_instance.perceptron(x,y)
# print(prediction,acc,err)

app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        load()
        if not model_instance or not normalization_instance or not one_hot_instance:
            return jsonify({"error": "Model or preprocessors not loaded"}), 500
        json_data = request.get_json()

        if not json_data or "user_data" not in json_data:
            return jsonify({"error": "Missing 'user_data' field in JSON"}), 400

        user_data = json_data["user_data"]

        if len(user_data) != 14:
            return jsonify({"error": f"Expected 14 features, got {len(user_data)}"}), 400

        ready_for_model = modify_user_input_for_network(user_data,one_hot_instance,normalization_instance)
        data = pd.read_csv(r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\training_data.csv",delimiter=";")
        x= data.iloc[:,:-1].values
        y = data.iloc[:,-1].values
        acc,error= model_instance.perceptron(x,y)
        prediction = round(model_instance.pred(ready_for_model[0])[0],3)
        print(prediction)
        preds= []
        for row in x:
            preds.append(round(model_instance.pred(row)[0]))
        print(preds)
        matrix = model_instance.confusion_matrix(preds,y.reshape(-1).tolist())
        print(matrix)
        matrix = matrix.values.tolist()
        #1 is leave 0 stays
        tn = matrix[0][0]  # True Negatives
        fp = matrix[0][1]  # False Positives
        fn = matrix[1][0]  # False Negatives
        tp = matrix[1][1]  # True Positives
        leave_acc = tp/ (tp + fn) if (tp + fn) > 0 else 1
        stay_acc = tn / (tn + fp) if (tn + fp) > 0 else 1
        chance = "low" if prediction<=0.20 else "average"  if prediction<=0.60 else "high" if prediction<=0.95 else "very high"


        # ##forwrd
        return jsonify({"prediction": prediction,
                        "accuracy":round(acc,5)*100,
                        "error_on_data":round(error,5)*100,
                        "leave_bank_acc":leave_acc,
                        "stay_in_bank_acc":stay_acc,
                        "chance":chance})
    except Exception as e:
        return f"<h1>Error: {str(e)}</h1>", 500


if __name__ == "__main__":
    app.run(debug=True)

