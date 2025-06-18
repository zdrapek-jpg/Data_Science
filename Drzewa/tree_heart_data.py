import numpy as np
import pandas as pd

data = pd.read_csv(r"C:\Program Files\Pulpit\Data_science\Zbiory\heart.csv")

cols = [col  for col in data.columns[:-1]]
target_col = data.columns[-1]

x = data.iloc[:,:-1]
y = data.loc[:,"target"]
print(data.shape)
print(data.info)
print(data.describe())
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree

from Data_Preprocessing.SPLIT_train_valid_test import   SplitData

this = SplitData()
this.set(train=0.6,valid=0.20,test=0.20)
x,y,x_,y_,x_test,y_test  = this.split_data(data)

print(x.shape)
print(y.shape)
print(x_.shape)
print(x_test.shape)
my_tree = DecisionTreeClassifier(criterion="entropy",random_state=42)

data_for_grid ={ "max_depth":np.arange(1,7), "min_samples_leaf" :  [i for i in range(3,10)]  }
# podajemy odpowiednio, drzewo z parametrami, parametry do grid searcha, ilość podziałów,po czym patrzymy jakość modelu
grid_search = GridSearchCV(my_tree,param_grid=data_for_grid,cv=6,scoring="accuracy")
grid_search.fit(x,y)
print(grid_search.best_params_,grid_search.best_score_)
best_params = list(grid_search.best_params_.values())
grid_search.fit(x_,y_)
print(grid_search.best_params_,grid_search.best_score_)
grid_search.fit(x_test,y_test)
print(grid_search.best_params_,grid_search.best_score_)
my_tree.min_samples_leaf= best_params[-1]
my_tree.max_depth = best_params[0]
my_tree.fit(x,y)
valid = my_tree.score(x_,y_)
tests = my_tree.score(x_test,y_test)
from sklearn.metrics import  accuracy_score
pred1 = my_tree.predict(x)
acc = accuracy_score(y,pred1)
print("accuracy: ",acc)
print("valid accuracy:",valid,"tests accuracy: ",tests)
import  matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
tre_plt= plot_tree(my_tree,feature_names=cols,class_names=data[target_col].unique().astype(dtype=str),filled=True)
plt.show()
