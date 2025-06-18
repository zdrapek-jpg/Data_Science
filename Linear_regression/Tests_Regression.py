from random import random

import pandas as pd

from Linear_reg_class import LinearRegression
from Gui_szkolenie_4.Data.Transformers import  Transformations,StandardizationType
##  Testy dla LinearRegression
import numpy as np
from sklearn.linear_model import LinearRegression as Linear

from sklearn.metrics import mean_squared_error
x = np.linspace(1,100,1000)
np.random.shuffle(x)
# każdy pkt ma 2 wsp
x2 =np.reshape(x,(500,2))
# każdy pkt ma 1 współrzędną
x1 =np.reshape(x,(1000,1))
x3 = np.reshape(x,(250,4))



zbiory = [x2,x3]

def create_data(x_data):
    shape_x = x_data.shape
    y_znak = np.random.choice(size=shape_x, a=[-2, 2])
    y_like_threshold = np.random.random(shape_x) * y_znak
    # print(y_like_threshold)
    y_val = 2*x_data -   y_like_threshold  # y  = x -1.5 * losowa  to wciąż macierz jak w w
    if y_val.shape[-1] == 1:
        return x_data.reshape(x.data.shape[0],),y_val.reshape(x_data.shape[0],)
    else:
        y_val = np.sum(y_val,axis=1)
    return x_data,y_val
#
datas = []
for x in zbiory:
    x,y =create_data(x)
    datas.append([x,y])
    data =pd.DataFrame(x)
    tf = Transformations(data,StandardizationType.Z_SCORE)
    x = tf.standarization_of_data(data).values

    lin = LinearRegression(x,y)
    print(lin.data_mean(x))
    print(lin.data_mean(y))

    ln_model = Linear()
    ln_model.fit(x,y)
    print("\n\n")
    print(ln_model.coef_)
    print(ln_model.intercept_)

#  check the output line with predictions according to the y =bx+a  values
#
# for data in datas:
#     error = 0
#     for i,(point_x,ygre) in enumerate(zip(datas[-1][0],datas[-1][1])):
#         y_pred = np.sum(point_x*lin.wsp_b)+lin.wsp_a
#
#         #print("pred :", y_pred , "  ", ygre," ")
#         # MSE
#         y_m = datas[-1][1].mean()
#         error+= (ygre-y_pred)**2

    #error/len(datas[-1][1])
    #print(error)

#       testy

data = pd.read_csv(r"C:\Program Files\Pulpit\Data_science\Zbiory\high_school_sat_gpa.csv",
                   sep=' ', usecols=['math_SAT','verb_SAT','high_GPA'])
# print(data.head())
# print(data.dtypes)
# print(data.info)
# print(data.size)
y_val  = data.loc[:,"verb_SAT"]
x_val = data.iloc[:,:-1]

 #normalize data

tf = Transformations(data)
data_ =tf.standarization_of_data(data)
print(data.shape)
print(data_.shape)
y_val  = data_.iloc[:, -1].values

x_val = data_.iloc[:, :-1].values

model = LinearRegression(x_val,y_val)
model.fit()

print("dla ocen:")
print(model.wsp_b,model.wsp_a)

#



    # średnie dla każdego punktu x i  y Sxy  =  (x-x_sr) i (y-y_sr)   x(100,3) (3,)
    #  pseudo inversja w np pinv()   y^*
    # scikit-learn dokumentacja standaryzacje->metody
    # pinv= (A^T * A)^-1*A^T
    #dodatkowo wyliczamy na x    Sxx= (x-x_sr)**2
    #następnie mamy
    #  y = ax+b
    #           SUMA(Sxy)
    #  b =      --------      = b jako skalar albo vektor licz np b = [2.3, 1.2, 3.5]
    #           SUMA(Sxx)
    #
    #  a  =    y_sr   -  b * x_sr      np  y_sr =[15.5]  b = [2.3, 1.2, 3.5]  x_sr  = [1, 2.1, 3.4]
    #   a =  y_sr    -  (2.3*1 + 1.2*2.1 + 3.4 *3.5)   a jest jedno niezlażnie od wymiarów x
    #   regresja logistyczna
    #                     1
    #       y   =     --------------
    #           1+ e -(a + (b0*x0 +b1*x1+ b2*x2) )
    #




ready = Linear()
ready.fit(x_val,y_val)
print("model")
print(ready.coef_)
print(ready.intercept_)


my = 0
skl = []
for i in range(data_.shape[0]):
    skl_ =model.predict(data_.iloc[i, :-1])
    my_ = ready.predict([data_.iloc[i, :-1]])
    skl.append(skl_)
    my+= (my_ - data_.iloc[i, :-1]) ** 2
skl_ = np.array(skl)
print("moj",end=" ")
print(my_ / data_.shape[0])
print("kupny",end=" ")
#print(mean_squared_error(data_[:, -1], skl_) / data_.shape[0])


from Ransac_random_sample_consensus import Ransac
## RANSAC
data = data.iloc[:,[0,-1]]
new_point = pd.DataFrame([[5.6, 1003],[8,385],[5.1,1080],[3.0,900],[5.7,780],[5,309],[6,1120]], columns=data.columns)

# Append the new row
data = pd.concat([data, new_point], ignore_index=True)
rs = Ransac(data.values,min_samples=0.3,max_iter=250,distance=340)
rs.fit(data.values,True)
print(rs.best_model)



