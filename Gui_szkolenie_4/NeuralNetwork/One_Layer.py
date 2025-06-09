import numpy as np
import random


class LayerModificationMetaClass(type):
    """Purpose of this class is to add attributes and their values especially for
    create fields"""
    def __new__(cls, name, bases, dct):
        original_init = dct.get('__init__')
        def new_init(self, len_data, wyjscie_ilosc=1, activation_layer=None, optimizer=None, gradients=None):
            original_init(self, len_data, wyjscie_ilosc, activation_layer, optimizer, gradients)
            # initialization based on optimizer

            self.weights_exponential_d = np.zeros((wyjscie_ilosc, len_data))
            self.biases_exponential_d = np.zeros((wyjscie_ilosc,))

            if self.optimizer in ["momentum", "adam","RMSprop"]:
                self.v_weights = np.zeros((wyjscie_ilosc,len_data))
                self.v_biases =     np.zeros(wyjscie_ilosc,)
                self.Beta1 = 0.95
            if self.optimizer in ["RMSprop","adam"]:
                self.epsilion = 1e-8
                self.Beta2 = 0.999
            if self.optimizer=="adam":
                self.timestep =1
                self.m_weights =np.zeros((wyjscie_ilosc,len_data))
                self.m_biases=   np.zeros(wyjscie_ilosc,)
                self.timestep=1

        dct['__init__'] = new_init
        return super().__new__(cls, name, bases, dct)


def optimizer_decorator(function):
    """
    Decorator to describe  logic before or after a function call.
    """
    def wrapper(self, *args, **kwargs):
        if self.optimizer == "adam":
            print("applying Adam optimization")
        if self.gradients =="batch":
            print("applying batch gradient descent")
        if  self.gradients == "mini-batch":
            print("applying mini batch gradient descent")
        elif self.optimizer == "momentum":
            print("applying Momentum optimization")
        elif self.optimizer=="RMSprop":
            print("applying RMSprop optimization")
        else:
            print("it seems that function uses Stochastic Gradient Descent optimization")
        print("Model Optimizer Information:")
        try:
            print("Optimizer:".ljust(15) + self.optimizer.ljust(10))
            print("Gradients:".ljust(15) + self.gradients.ljust(10))
            print(f"Alpha: {str(self.alfa): ^15}")
        except:
            pass

        print("Activation Layer:".ljust(20, " "), str(self.activation_layer).ljust(10," "))
        print("Weights shape:".ljust(20," "), str(list(self.wagi.shape)).ljust(10," "))
        print("Biases shape:".ljust(20," "), str(list(self.bias.shape)).ljust(10," "))
        print("-" * 30)
        return function(self, *args, **kwargs)

    return wrapper




class LayerFunctions(metaclass=LayerModificationMetaClass):
    __slots__ = ["len_data","wyjscia_ilosc","activation_layer","bias","wagi","alfa","loss","accuracy","Beta1","weights_exponential_d","biases_exponential_d","v_weights","v_biases","optimizer","gradients","epsilion","Beta2","moment","m_weights","m_biases","timestep"]
    def __init__(self, len_data:int, wyjscie_ilosc:int =1,activation_layer:str=None,optimizer:str="",gradients:str=None ):
        """
        :param len_data:  data input lenght of one row  int
        :param wyjscie_ilosc:  number of neurons we want to have int
        :param activation_layer:  name of activation layer sigmoid,relu,elu,
        :param optimizer:  optional : adam,RMSprop,momentum
        :param gradients:  optional : mini-batch,batch, else SGD
        """
        self.len_data = len_data
        self.wyjscia_ilosc = wyjscie_ilosc
        self.activation_layer = activation_layer
        self.optimizer = optimizer
        self.gradients = gradients

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

            pochodna_wyjscia =y_pred-y_origin
            ### warstwy ukryte
            if self.activation_layer == "softmax":
                gradient = pochodna_wyjscia
                self.biases_exponential_d =gradient
                self.weights_exponential_d = np.outer(gradient,self.alfa)*point
                self.backward_update_params()
                return
            pochodna_aktywacji = self.derivations(y_pred)


            gradient  = pochodna_wyjscia* pochodna_aktywacji

            self.biases_exponential_d = gradient
            self.weights_exponential_d= gradient*point.reshape(1,self.len_data)
            self.backward_update_params()
            return gradient
        pochodna_aktywacji = self.derivations(y_pred)

        # gradient dla wszystkoch warstw ukrytych
        gradient =   np.dot(weights_forward.T,gradient2)
        gradient *= pochodna_aktywacji
        self.biases_exponential_d = gradient
        self.weights_exponential_d= np.outer(gradient,point)*point
        self.backward_update_params()
        return gradient

    def backward_batches(self,y_pred=None,point=None,pochodna_wyjscia=None,weights_forward=None,gradient2=None,for_average=None):
        pochodna_aktywacji = self.derivations(y_pred)

        ## obiczanie gradientu w pierwszej warstwie od konca
        if pochodna_wyjscia is  not None:
            gradient = pochodna_wyjscia * pochodna_aktywacji
            self.biases_exponential_d   +=   gradient
            self.weights_exponential_d  +=  gradient * point.reshape(1, self.len_data)
            return gradient

        if weights_forward is not  None:
            # gradient dla wszystkoch warstw ukrytych
            gradient = np.dot(weights_forward.T, gradient2)
            gradient *= pochodna_aktywacji
            self.biases_exponential_d  +=  gradient
            self.weights_exponential_d  += np.outer(gradient, 1) * point
            return  gradient

    def backward_update_params(self,for_average=1):
        self.weights_exponential_d /= for_average
        self.biases_exponential_d /= for_average

        if self.optimizer=="momentum":
            self.v_weights = self.Beta1 * self.v_weights+ (1 - self.Beta1) * self.weights_exponential_d
            self.v_biases = self.Beta1 * self.v_biases + (1 - self.Beta1) * self.biases_exponential_d

            self.bias -= self.alfa *self.v_biases
            self.wagi-= self.alfa*self.v_weights
            return
        if self.optimizer=="RMSprop":
            self.v_weights = self.Beta2 * self.v_weights + (1 - self.Beta2) * (self.weights_exponential_d ** 2)
            self.v_biases = self.Beta2 * self.v_biases + (1 - self.Beta2) * (self.biases_exponential_d ** 2)

            self.wagi -= self.alfa / (np.sqrt(self.v_weights) + self.epsilion) * self.weights_exponential_d
            self.bias -= self.alfa / (np.sqrt(self.v_biases) + self.epsilion) * self.biases_exponential_d
            return
        if self.optimizer=="adam":
            self.m_weights =self.Beta1*self.m_weights+ (1-self.Beta1)*self.weights_exponential_d
            self.m_biases  =self.Beta1*self.m_biases +(1-self.Beta1)*self.biases_exponential_d
            self.v_weights =self.Beta2*self.v_weights + (1-self.Beta2) * (self.weights_exponential_d**2)
            self.v_biases  =self.Beta2*self.v_biases +(1-self.Beta2)*(self.biases_exponential_d**2)

            t = getattr(self, "timestep", 1)
            self.timestep = t + 1
            m_hat_w = self.m_weights / (1 - self.Beta1 ** t)
            m_hat_b = self.m_biases / (1 - self.Beta1 ** t)

            v_hat_w = self.v_weights / (1 - self.Beta2 ** t)
            v_hat_b = self.v_biases / (1 - self.Beta2 ** t)
            self.wagi -=self.alfa*m_hat_w/(np.sqrt(v_hat_w)+self.epsilion)
            self.bias -= self.alfa*m_hat_b/(np.sqrt(v_hat_b)+self.epsilion)
        else:
            self.wagi-=self.alfa * self.weights_exponential_d
            self.bias-= self.alfa * self.biases_exponential_d

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
                            np.where(suma_wazona < 0, self.alfa * (np.exp(suma_wazona) - 1), 0))
        if self.activation_layer == "relu":
            return np.maximum(0, suma_wazona)
        if self.activation_layer =="softmax":
            dul= sum(suma_wazona)
            return suma_wazona/dul
        else:
            raise "brak zdefiniowanej funkcji aktywującej "

    def derivations(self, y_pred):
        """
        :return  derivative of predicted output to activation function
        """
        if self.activation_layer == "sigmoid":
            return y_pred * (np.ones_like(y_pred) - y_pred)

        if self.activation_layer == "elu":
            alpha = self.alfa
            return np.where(y_pred >= 0, y_pred, -alpha * np.exp(y_pred))

        if self.activation_layer == "relu":
            return np.where(y_pred >= 0, y_pred, 0)


    def start(self, alfa=None,seed=32):
        """
        :param seed:  seed for weights biases
        :param alfa is eta/alpha/learning rate we can give
           intialize parameters of layer  [alfa,bias, weight with  SHAPE = (self.len_data,wyjscia_ilosc).T]
           :return alfa
           """
        self.random_alfa(alfa)
        self.random_bias(seed=seed)
        self.random_weights(seed=seed)
        return self.alfa[0]

    def random_weights(self,seed=32):
        np.random.seed(seed)
        self.wagi = np.random.uniform(low=-0.2,high=0.2,size=(self.wyjscia_ilosc,self.len_data))

    def random_bias(self,seed=32):
        np.random.seed(seed)
        self.bias = np.random.uniform(low=-0.2,high=0.2,size=(self.wyjscia_ilosc,))

    def random_alfa(self,a=None):
        if a!=None:
            self.alfa = np.array([a])
        else:
            x =np.array([0.05])


    @optimizer_decorator
    def return_params(self,show=False):
        """
        Returns model parameters.
        """
        if show:
            return {
            "wagi": self.wagi,
            "bias": self.bias,
            "activation": self.activation_layer}
        return
