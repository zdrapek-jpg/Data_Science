import matplotlib.pyplot as plt
import numpy as np


def draw_points(x,labels,count_categories=None,point_for_classification=None,colorized=True,lims =[[0,10],[0,10]]):
    if colorized:
        cmap = plt.get_cmap("tab10", count_categories)  # Categorical colormap

        plt.figure(figsize=(5,6))
        labels =labels.tolist()
        for point,label in zip(x,labels):

            plt.scatter(point[0],point[1],color =cmap(int(label)),s=45,alpha=0.7)
            plt.annotate(f"{labels.index(label)}",(point[0],point[1]),
                         textcoords ="offset points",xytext=(0,4.5),fontsize =7,color= cmap(label),ha="center")

    else:
        plt.figure(figsize=(5, 6))
        labels = labels.tolist()
        for point, label in zip(x, labels):
            plt.scatter(point[0], point[1], s=45, alpha=0.7)
            plt.annotate(f"{labels.index(label)}", (point[0], point[1]),
                         textcoords="offset points", xytext=(0, 4.5), fontsize=7, ha="center")
    if point_for_classification is not None:
        plt.scatter(point_for_classification[0], point_for_classification[1], color="orange", alpha=1, s=89)
        plt.annotate(F"x", (point_for_classification[0], point_for_classification[1]), textcoords="offset points",
                     xytext=(0, 4), fontsize=12, ha="center")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.xlim(lims[0])
    plt.ylim(lims[1])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
    """
    :return draw points linear interpertation and colour inliners and outliners
    :param data = matrix of points like [x,y]
    :param mask [true,false] matrix defining if its inlier or outlier
    :param equasion (a,b)  returns linear like: y = a*x +b     
    """
import numpy as np
def draw_points_with_line(data,mask,equasion,second =None,third =None):
    a,b = equasion
    a = a[0]
    min_x = data[:,0].min()
    max_x = data[:,0].max()
    min_y = data[:,-1].min()
    max_y =data[:-1].max()

    plt.figure(figsize=(10,10))
    x = np.linspace(min_x-3,max_x+3,100)
    y= a * x +b
    #print(x)
    #print(y)
    plt.plot(x, y, c="blue",label ="Ransac model")
    X =data[:,0]
    Y = data[:,-1]
    if mask is None:
        raise ValueError("Error: 'mask' cannot be None.")

    # Ensure mask shape matches data
    if mask.shape[0] != X.shape[0]:
        raise ValueError(f"Error: mask shape {mask.shape} does not match data shape {X.shape}")

    plt.scatter(X[mask],Y[mask],c="green",marker=".",label="Inliers",s=25)
    plt.scatter(X[~mask], Y[~mask], c="red", marker=".", label="Outliers",s=30)
    if second is not None:
        a,b = second
        y_ = a[0]*x +b
        plt.plot(x,y_,c="orange",label="Regression")
    if third is not None:
        a1,b1 = third
        y1 = a1[0] * x + b1
        plt.plot(x, y1, c="purple", label="Scikit_rans")





    plt.xlabel("x_val")
    plt.ylabel("y_val")
    plt.xlim(min(x)-1,max(x)+1)
    plt.ylim(min(y)-5,max(y)+5)
    plt.legend()
    plt.show()





