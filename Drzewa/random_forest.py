from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X,y = make_blobs(n_samples=500,centers=5,random_state=100,cluster_std=0.6,center_box=(0,10))
#plt.scatter(X[:,0],X[:,1],c=y,s=30,cmap="viridis")
#plt.show()
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import plot_tree
clasifier = DecisionTreeClassifier(max_depth=4,criterion="entropy",max_leaf_nodes=6)
clasifier.fit(X,y)
#plot_tree(decision_tree=clasifier,filled=True,class_names=["0","1","2","3","4"],feature_names=["0","1"])
#plot_decision_regions(X,y,clasifier)

# uczenie zespołowe lasy drzew decyzyjnych
# n_estimators is the count of trees that are built and max_samles= amout of data for one tree
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion="entropy",n_estimators=20,max_samples=0.8)
forest.fit(X,y)
print(forest.score(X,y))
plot_decision_regions(X, y, clf=forest)