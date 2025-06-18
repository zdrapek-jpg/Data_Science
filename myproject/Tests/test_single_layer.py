#testy czy struktura warstwy neuronowej działa poprawnie
import numpy as np
from NeuralNetwork.One_Layer import LayerFunctions


### obiekt do podziału z x i y w sobie razem jako jeden obiekt
#to co zwracamy jest już podzielne na x i y walidacyjne, testowe i treningowe

functions = ["sigmoid","Elu","Relu"]
def initialization_tests(inputs,activation,alpha_):
    layer = LayerFunctions(len_data=inputs,wyjscie_ilosc=inputs*3)
    layer.activation_layer= activation
    layer.start(alfa=alpha_)
    return layer.activation_layer,layer.alfa,layer.wagi,layer.bias,layer.len_data,layer.wyjscia_ilosc
#
# for init_params in range(3,4):
#     for act_f in functions:
#         alpha_ = init_params / random.choice([10, 100, 1000])
#         act,alfa,wagi,biasy,dlugosc,wyjscia =initialization_tests(init_params,act_f,alpha_=alpha_)
#         #assercje sprawdzające parametry funkcji czy są ustawiane takie jakimi je przekazujemy
#         assert(act in functions), f"there is no activation function {act}"
#         print(alpha_,alfa,wagi,act)
#         assert(alfa[0] ==alpha_), f" alfa nie jest ustawiana parametrem z zewnątrz {alpha_} !={alfa[0]}  "
#         assert wagi.shape[1]==init_params,  f"parametr wag niezgodny z iloscią wejść\n wagi:{wagi.shape[0]}\nwejscia{init_params}"
#         assert len(biasy)==init_params*3, f"\n{biasy}:{len(biasy)}\n  {init_params*3} "
#         assert dlugosc == init_params
#         assert wyjscia == init_params*3

def init(x,y):

    for act in ["sigmoid","elu","relu"]:
        layer = LayerFunctions(4,1,activation_layer=act,optimizer="",gradients="")
        layer.start(0.2)

        layer.wagi = np.array([[0,1,0,1]],dtype=float)
        layer.bias = np.array([0],dtype=float)
        prd_forw =layer.train_forward(x)
        print("output: ",prd_forw)
        grad =layer.backward_sgd(y_pred=prd_forw,point=x,y_origin=y)
        print("gradient: ", grad)
        print(layer.wagi)
        prd_forw = layer.train_forward(x)
        print("after",prd_forw)

    return prd_forw


def check(point,y):
    layer = LayerFunctions(4, 2, activation_layer="softmax", optimizer="", gradients="")
    layer.start(0.6)

    layer.wagi = np.array([[1, 0, 0, 1],[1,1,1,0]], dtype=float)
    layer.bias = np.array([0,0], dtype=float)

    pred =layer.train_forward(point)
    print(pred)
    layer.backward_sgd(y_pred =pred,point=point,y_origin=y)
    print(layer.train_forward(point))


### data
x= np.array([[1,2,0,0],[0,0,1,2],[0,0,1,0]],dtype=float)
y = np.array([1,0],dtype=float)
print(check(x[0],np.array([0,1])))
