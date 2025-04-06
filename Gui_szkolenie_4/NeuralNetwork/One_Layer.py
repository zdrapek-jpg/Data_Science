import numpy as np
import random



class LayerFunctions:
    def __init__(self, len_data, wyjscie_ilosc=1,optimizer=None,activation_layer=None ):
        self.len_data = len_data
        self.wyjscia_ilosc = wyjscie_ilosc
        self.optimizer = optimizer
        self.activation_layer = activation_layer
        self.bias = None
        self.wagi = None
        self.prog = 0.5
        self.alfa = 0.09
        self.one_hot_encoded = None
        self.loss = []
        self.accuracy = []
        self.next = None

    def train_forward(self, point):
        suma_wazona = self.forward(point)
        outputs = self.activation(suma_wazona)
        # print(suma_wazona,outputs)
        return outputs

    def forward(self, point):
        point = np.array(point)
        PROD = np.dot(self.wagi, point)
        return PROD + self.bias

    def backward(self,y_pred,point=None,y_origin=None,weight_forward=None,gradient2=None):
        ### pierwszy wariant wyliczamy gdy warstw jest warstwą w środku
        # warunek do lini przekazujemy gradient poprzedniej warstwy i wagi poprzedniej warstwy
        if gradient2 is not None and weight_forward is  not None:
            # Mnożenie macierzowe, aby obliczyć gradient dla bieżącej warstwy z wag
            gradient_activation = self.derivations(y_pred,y_origin)
            gradient = np.dot(np.transpose(weight_forward), gradient2)
            gradient *= gradient_activation
            self.bias -= self.alfa * gradient
            weight_update = np.zeros_like(self.wagi)
            for i in range(self.wagi.shape[0]):
                # alfa * deltawag * wejśćie  czyli poprzednie wyjścia lub punkty wchodzące do sieci
                weight_update[i] = self.alfa[i] * gradient[i] * point
            self.wagi -= weight_update
            return gradient

        else:
            gradient = self.derivations(y_pred, y_origin)
            self.bias -= self.alfa * gradient
            weight_update = np.zeros_like(self.wagi)
            for i in range(self.wagi.shape[0]):
                # alfa * deltawag * wejśćie  czyli poprzednie wyjścia lub punkty wchodzące do sieci
                weight_update[i] = self.alfa[i] * gradient[i] * point
            self.wagi -= weight_update
            return gradient

    def softmax(cls, suma_wazona):
        return np.exp(suma_wazona) / np.sum(np.exp(suma_wazona))

    def activation(self, suma_wazona):  # [macierz wynikowa jednego wymiary tyle ile jest neuronów]
        if self.activation_layer == "softmax":
            return self.softmax(suma_wazona)
        if self.activation_layer == "sigmoid":
            z = lambda x: 1 / (1 + np.exp(-x))
            return z(suma_wazona)
        if self.activation_layer == "Elu":
            return np.where(suma_wazona > 0, suma_wazona,
                            np.where(suma_wazona < 0, 0.01 * (np.exp(suma_wazona) - 1), 0))
        if self.activation_layer == "Relu":
            return np.maximum(0, suma_wazona)

    def derivations(self, y_pred, y_origin):
        if self.activation_layer == "softmax":
            pass

        if self.activation_layer == "sigmoid":
            return y_pred * (np.ones_like(y_pred) - y_pred)

        if self.activation_layer == "Elu":
            alpha = self.alfa
            return np.where(y_pred >= 0, 1, alpha * np.exp(y_pred))

        if self.activation_layer == "Relu":
            return np.where(y_pred >= 1, 1, 0)

    def start(self, alfa=None):
        self.random_alfa(alfa)
        self.random_bias()
        self.random_prog()
        self.random_weights()
        return self.alfa[0]

    def random_weights(self):
        self.wagi = np.random.rand( self.len_data,self.wyjscia_ilosc).T
        # self.m = np.zeros_like(self.wagi)
        # self.v = np.zeros_like(self.wagi)
    def random_bias(self):
        self.bias = np.array([round(random.random() * random.choice([0.9, 0.9, 0.8, 0.8]), 3)] * self.wyjscia_ilosc).T
    def random_alfa(self,a=None):
        if a!=None:
            self.alfa = [a]*self.wyjscia_ilosc
        else:
            x =round(random.random() * 0.65, 3)
            self.alfa = [round(x if x<=0.2 and x>=0.09  else x+0.05 if x<=0.09 else x-0.2,4) ] * self.wyjscia_ilosc
    def random_prog(self):
        self.prog = np.array([round(random.random() * 1.1, 3)] * self.wyjscia_ilosc)

