from matplotlib.colors import ListedColormap
from Data_Preprocessing.SPLIT_train_valid_test import   SplitData
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


data = pd.read_csv(r"C:\Program Files\Pulpit\Data_science\Zbiory\iris_orig.csv",names = ["x1","x2","x3","x4","y"])
data = data.iloc[:,[0,3,4]]
cords_factor = data.loc[:,"y"]
data["y"] = pd.Series(pd.factorize(cords_factor)[0], index=data.index)

this =SplitData()
this.set(train=0.5,valid=0.3,test=0.2)
x,y,x_,y_,x_test,y_test  = this.split_data(data)
tree_cls= DecisionTreeClassifier(max_depth=3,criterion="entropy",random_state=42)
tree_cls.fit(x,y)
#print(tree_cls.get_depth())
pred = tree_cls.predict(x_)



import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(12, 6))
plot_tree(tree_cls, feature_names=[ "x3", "x4"], class_names=["0", "1", "2"], filled=True)
plt.show()
import numpy as np


x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Predict class for each point in the mesh grid
Z = tree_cls.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Define colors
cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
cmap_bold = ["red", "green", "blue"]

# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

# Scatter plot of the actual data points
scatter = plt.scatter(x[:, 0], x[:, 1], c=y, cmap=ListedColormap(cmap_bold), edgecolor="k")

# Add labels and title
plt.xlabel("x0")
plt.ylabel("x4")
plt.title("Decision Tree Decision Boundaries")

# Show legend
legend_labels = {0: "Class 0", 1: "Class 1", 2: "Class 2"}
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap_bold[i], markersize=8, label=legend_labels[i]) for i in range(3)]
plt.legend(handles=handles, title="Classes")
plt.show()

cls = {}
for point,clas in zip(x,y):
    if clas in cls.keys():
        cls[clas]+=1
    elif clas not in cls.keys():
        cls[clas]=1
klasy = cls.values()
g = 1
for num in klasy:
    g-=(num/x.shape[0])**2

from sklearn.datasets import make_moons,make_blobs
raw_data = make_moons(n_samples=851,noise=0.16)
data = raw_data[0]
target = raw_data[1]
data = pd.DataFrame(data,columns=["x1","x2"])
data["y"]= target.reshape(-1,1)
print(data)


from Data_Preprocessing.SPLIT_train_valid_test import SplitData
spl = SplitData()
spl.set(0.5,0.25,0.25)
x,y,x_,y_,x_test,y_test = spl.split_data(data)
print(x.shape,y.shape,x_.shape,y_.shape,x_test.shape,y_test.shape)
plt.scatter(x[:,0],x[:,1],c=y,cmap="RdYlBu",label ="training set")
plt.scatter(x_[:,0],x_[:,1],c=y_,cmap ="RdYlBu",marker="x",label = "validating set")
plt.scatter(x_test[:,0],x_test[:,1],c=y_test,cmap ="RdYlBu",marker ="*",label="tests points")
plt.show()

from sklearn.tree import DecisionTreeClassifier,plot_tree
tree_cls = DecisionTreeClassifier(criterion="entropy",random_state=42,max_depth=5,min_samples_leaf=4,max_leaf_nodes=10)
tree_cls.fit(x,y)
print(tree_cls.score(x_test,y_test))
print(tree_cls.score(x_,y_))
print(tree_cls.score(x,y))
plt.figure(figsize=(12, 6))
plot_my_tree = plot_tree(tree_cls,feature_names=["x1","x2"],class_names=["0","1"] ,filled=True)

x_min,x_max = x.min()-0.2,  x.max()+0.2
y_min,y_max = y.min()-1 , y.max()+1
xx,yy = np.meshgrid(np.linspace(x_min,x_max,100),np.linspace(y_min,y_max,100))
Z = tree_cls.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)
cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
cmap_bold = ["red", "green", "blue"]

# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

# Scatter plot of the actual data points
scatter = plt.scatter(x[:, 0], x[:, 1], c=y, cmap="RdYlBu",s=10)
scatter = plt.scatter(x_[:, 0], x_[:, 1], c=y_, cmap="RdYlBu",marker="x")
scatter = plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap="RdYlBu",marker="^")
plt.show()


# grid search szuka najlepszego podziału w drzewie
from sklearn.model_selection import GridSearchCV
grid_parameters = {"max_depth":np.arange(3,10),
                   "min_samples_leaf": [i for i in range(2,8)]}
# cv to ilość podziałów
gs = GridSearchCV(tree_cls,grid_parameters,scoring="accuracy",cv=5)
gs.fit(x,y)
print(gs.best_estimator_)
print(gs.best_score_)
print(gs.best_params_)



