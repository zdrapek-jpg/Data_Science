# Tu Będzie zbudowana sieć neuronowa która edzie miałą warstwy złożone z One_Layer


import numpy as np
from One_Layer import LayerFunctions

class NNetwork():
    def __init__(self,w=None,epoki=None,alpha=0.02,batche = None):
        self.epoki = epoki
        self.alpha = alpha
        if epoki is None:
            self.epoki = 2

        self.LineOne = LayerFunctions(len_data=30,wyjscie_ilosc=4,activation_layer="Relu")
        self.LineTwo = LayerFunctions(len_data=4,wyjscie_ilosc=1,activation_layer="sigmoid")
        self.LineOne.start(self.alpha)
        self.LineTwo.start(self.alpha)
        # if w is not None:
        #     self.LineTwo.wagi = w
        #     self.LineOne.bias = np.array([0.5,0.5,0.5])
        #     self.LineTwo.bias = np.array([0.5,0.5,0.5])

        self.loss = []
        self.accuracy = []

        if batche=="batch":
            self.LineTwo.optimizer="batch"
            self.LineOne.optimizer="batch"
