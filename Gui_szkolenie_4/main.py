from Data.load_user_data import modify_user_input_for_network
from NeuralNetwork.Training_structre import training
from NeuralNetwork.getData import data_preprocessing
import sys

# załadowanie i przetworzenie danych do modelu

data =data_preprocessing()
# stworzenie modelu i szkolenie go na danych data
training(data)


# załadowanie danych wpisanych przez pracownika w gui lub w oknie webowym
#np:
user_data = ["Henryk",619,"France","Female",2,1,1,101348.88,58,"management","married","tertiary",6429,"no"]


