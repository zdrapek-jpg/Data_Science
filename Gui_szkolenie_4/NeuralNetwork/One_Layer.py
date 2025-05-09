import numpy as np
import random



class LayerFunctions:
    __slots__ = ["len_data","wyjscia_ilosc","activation_layer","bias","wagi","alfa","loss","accuracy","Beta","weights_exponential_d","biases_exponential_d","v_weights","v_biases","optimizer","gradients","epsilion","RMSprop"]
    def __init__(self, len_data, wyjscie_ilosc=1,activation_layer=None,optimizer=None,gradients=None ):

        self.len_data = len_data
        self.wyjscia_ilosc = wyjscie_ilosc

        self.activation_layer = activation_layer
        self.bias = None
        self.wagi = None
        self.alfa = 0.009
        #self.one_hot_encoded = None
        # korekta pierwszego rzędu dla 1 parametru momentum
        self.optimizer =optimizer
        self.gradients=gradients
        if self.gradients=="batch" or self.gradients=="mini-batch":
            self.weights_exponential_d = None
            self.biases_exponential_d = None
        if self.optimizer=="momentum":
            self.Beta =0.9
            #exponentially weighted averages of derivatives
            self.v_weights = None
            self.v_biases = None
        if self.optimizer=="adagrad":
            self.epsilion = None
        if self.optimizer=="adam":
            self.Beta=0.9
            self.RMSprop = 0.95
            self.epsilion = None

    def train_forward(self, point):
        """
           :param  x row like structure in np.array
           :return  activation of product
           """
        suma_wazona = self.forward(point)
        outputs = self.activation(suma_wazona)
        # print(suma_wazona,outputs)
        return outputs
    def forward(self, point):
        """
           :param point x row
           :return product of weights * point +bias
           """
        PROD = np.dot(self.wagi, point)
        return PROD + self.bias


    def backward_sgd(self,y_pred,point=None,y_origin=None,weights_forward=None,gradient2=None):

        """
              Perform the backward pass (backpropagation) for a single neural network layer.

              Parameters:
              - y_pred (np.ndarray): Predicted output from the current layer.
              - point (np.ndarray, optional): Input to this layer (from previous layer or input features).
              - y_origin (np.ndarray, optional): Ground truth output (used at the output layer).
              - weights_forward (np.ndarray, optional): Weights from the next layer (used for hidden layers).
              - gradient2 (np.ndarray, optional): Gradient from the next layer (used for hidden layers).

              Returns:
              - gradient (np.ndarray): Computed gradient for this layer, to be used by the previous layer.
              """
        ## obiczanie gradientu w pierwszej warstwie od konca
        if weights_forward is None and  gradient2 is None:
            pochodna_aktywacji = self.derivations(y_pred)
            pochodna_wyjscia =y_pred-y_origin

            gradient  = pochodna_wyjscia* pochodna_aktywacji
            self.bias  -= self.alfa*gradient
            self.wagi  -= self.alfa*gradient*point.reshape(1,12)
            return gradient
        pochodna_aktywacji = self.derivations(y_pred)

        # gradient dla wszystkoch warstw ukrytych
        gradient =   np.dot(weights_forward.T,gradient2)
        gradient *= pochodna_aktywacji
        self.bias -= self.alfa*gradient
        self.wagi -= np.outer(gradient,self.alfa)*point
        return gradient

    def backward_batches(self,y_pred=None,point=None,pochodna_wyjscia=None,weights_forward=None,gradient2=None,for_average=None):
        pochodna_aktywacji = self.derivations(y_pred)

        ## obiczanie gradientu w pierwszej warstwie od konca
        if pochodna_wyjscia is  not None:
            gradient = pochodna_wyjscia * pochodna_aktywacji
            self.biases_exponential_d   +=   gradient
            self.weights_exponential_d  +=  gradient * point.reshape(1, 12)
            return gradient

        if weights_forward is not  None:
            # gradient dla wszystkoch warstw ukrytych
            gradient = np.dot(weights_forward.T, gradient2)
            gradient *= pochodna_aktywacji
            self.biases_exponential_d  +=  gradient
            self.weights_exponential_d  += np.outer(gradient, 1) * point
            return  gradient

        self.weights_exponential_d/=for_average
        self.biases_exponential_d /=for_average
        self.v_weights = self.Beta * self.v_weights + (1 - self.Beta) * self.weights_exponential_d
        self.v_biases = self.Beta * self.v_biases + (1 - self.Beta) * self.biases_exponential_d

        self.bias -= self.alfa *self.v_biases
        self.wagi-= self.alfa* self.v_weights
    def activation(self, suma_wazona):
        """
        :return activatiob of product according to choosen self.activation_layer
        """
        # [macierz wynikowa jednego wymiary tyle ile jest neuronów]
        if self.activation_layer == "sigmoid":
            z = lambda x: 1 / (1 + np.exp(-x))
            return z(suma_wazona)
        if self.activation_layer == "elu":
            return np.where(suma_wazona > 0, suma_wazona,
                            np.where(suma_wazona < 0, 0.01 * (np.exp(suma_wazona) - 1), 0))
        if self.activation_layer == "relu":
            return np.maximum(0, suma_wazona)

    def derivations(self, y_pred):
        """
        :return  derivative of predicted output to activation function
        """
        if self.activation_layer == "sigmoid":
            return y_pred * (np.ones_like(y_pred) - y_pred)

        if self.activation_layer == "elu":
            alpha = self.alfa
            return np.where(y_pred >= 0, 1, alpha * np.exp(y_pred))

        if self.activation_layer == "relu":
            return np.where(y_pred >= 0, y_pred, 0)

    def start(self, alfa=None):
        """
        :param alfa is eta/alpha/learning rate we can give
           intialize parameters of layer  [alfa,bias, weight with  SHAPE = (self.len_data,wyjscia_ilosc).T]
           :return alfa
           """
        self.random_alfa(alfa)
        self.random_bias()
        self.random_weights()
        if self.gradients == "batch" or self.gradients == "mini-batch":
            self.weights_exponential_d = np.zeros_like(self.wagi)
            self.biases_exponential_d = np.zeros_like(self.bias)
        if self.optimizer == "momentum":
            self.v_weights = np.zeros_like(self.wagi)
            self.v_biases = np.zeros_like(self.bias)
        if self.optimizer=="adagrad":
            self.epsilion= 1e-9

        return self.alfa[0]


    def random_weights(self):
        self.wagi = np.random.rand( self.len_data,self.wyjscia_ilosc).T*random.choice([0.2,-0.3,-0.2,0.3])
    def random_bias(self):
        self.bias = np.random.rand(self.wyjscia_ilosc).T*random.choice([0.1,-0.1,-0.2,0.2])
    def random_alfa(self,a=None):
        if a!=None:
            self.alfa = np.array([a])
        else:
            x =round(random.random() * 0.65, 3)
            self.alfa = np.array([round(x if x<=0.2 and x>=0.09  else x+0.05 if x<=0.09 else x-0.2,4) ])
    def return_params(self):
        return {"wagi":self.wagi,"bias":self.bias,"activation":self.activation_layer}
    def load_params(self):
        pass
