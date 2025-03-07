import pandas as pd
import numpy as np

from Algorytmy_minimalno_odleglosciowe.AANG_similarity.AANG import AANG
from Algorytmy_minimalno_odleglosciowe.K_nn.KnnClass import Knn
from Data_Preprocessing.draw_2d_arrays_with_labels import draw_points
from Data_Preprocessing.Multiple_points import multiply
from Data_Preprocessing.Transformers import Transformations

x = [[2.1,3.8],[4.5,3.1],[1.3,2.8],[2.1,1.0],[3.2,1.6],[6,6]]
y = [1,0,1,0,2,2]

data_frame = pd.DataFrame(x,columns=[f"x{i}" for i in range(len(x[0]))])
data_frame["y"]= y

multiply_data = multiply(data_frame, 20, 0.75)
#print(multiply_data)

y = multiply_data.loc[:, 'y']
x = multiply_data.iloc[:, :-1]
#print(data)

#normalizacja danych
norma = Transformations(x)
x = norma.standaryzacja_mean_score(x,y)


y = multiply_data.loc[:, 'y'].values
x = multiply_data.iloc[:, :-1].values

## AANG algorithm

# Instantiate the AANG model
aang_model = AANG(x, y,10)

# Build the AANG graph
graph = aang_model.build_aang_graph()

# Propagate activation and get predictions
predictions = aang_model.propagation_activation(graph)
print("Predictions:", predictions)

# Classify a new point
new_point = np.array([4.39, 6])
predicted_class = aang_model.classify_point(new_point, graph)
print("Predicted class for the new point:", predicted_class)

# Visualize the points and classification
print(draw_points(x, y, 3, new_point))


### knn

point_for_classification = np.array([4.39,6])

for i in range(2,20):
    d = Knn(data_frame, point_for_classification, p=2)
    d.classify_point_at_k_neightbours(x,y,point_for_classification,i)
    print(d.classes)

