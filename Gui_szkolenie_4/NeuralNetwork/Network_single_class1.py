# Tu Będzie zbudowana sieć neuronowa która edzie miałą warstwy złożone z One_Layer
import json
import random
import numpy as np
import pandas as pd
from Data.SPLIT_test_valid_train import SplitData
from NeuralNetwork.One_Layer import LayerFunctions
from Data.Decorator_time_logging import log_execution_time
from numpy import array
class NNetwork():
    __slots__ = ["epoki","alpha","_Network","train_loss","train_accuracy","valid_accuracy", "valid_loss","optimizer","gradients"]
    def __init__(self,epoki:int=None,alpha:float=0.02,optimizer:str=None,gradients:str=None):
        """
        :param epoki:  number of max iteration of model
        :param alpha:  learning rate best scores 0.01~0.0001
        :param optimizer:  optional : adam,RMSprop,momentum
        :param gradients:  optional : mini-batch,batch, else SGD
        """
        self.epoki = epoki
        self.alpha = alpha
        if epoki is None:
            self.epoki = 2
        #Przechowanie warstw jako lista oraz utrzymanie struktury sieci jako listy
        self._Network = []
        self.optimizer = optimizer
        self.gradients=gradients


        self.train_loss = []
        self.train_accuracy = []

        self.valid_accuracy = []
        self.valid_loss = []

    def add_layer(self,inputs:int,outputs:int,activation_layer_name:str,seed_for_all:int=32):
        """
        add layer to network one after another if shape of input size  is not accurate with input of new layer raise exception
        :param seed_for_all:  seed for bias and weights of this layer
        :param inputs: size of input data
        :param outputs: num of neurons
        :param activation_layer_name:  activation function for the layer
        :return:  add layer to model
        """
        # instancja klasy warstwy
        instance_of_layer = LayerFunctions(len_data= inputs,wyjscie_ilosc= outputs,activation_layer=activation_layer_name,optimizer=self.optimizer,gradients=self.gradients)
        instance_of_layer.start(self.alpha,seed=seed_for_all)


        if len(self._Network)>=1:
            if inputs!= self._Network[-1].wyjscia_ilosc:
                raise f"{inputs}!= {self._Network[-1].wyjscia_ilosc} powinny być takie same"
        self._Network.append(instance_of_layer)

    def train_sgq(self,x_train,y_train,x_validate,y_validate):
        """

        :param x_train: train data numpy.array()
        :param y_train:  y data numpy.array()
        :param x_validate: data for validation of learning process data numpy.array()
        :param y_validate: data for validation of learning process data numpy.array()
        :return: update parameters of network like weights and biases
        """
        for j in range(1, self.epoki + 1):
            error = 0
            for i, point in enumerate(x_train):
                ## forward pass
                outputs =[point]
                output = point
                for layer in self._Network:
                    output = layer.train_forward(output)
                    outputs.append(output)

                # cross entropy error
                error += (y_train[i]-outputs[-1][0])**2

                for  numer_warstwy in reversed(range(len(self._Network))):
                    warstwa = self._Network[numer_warstwy]
                    predykcja = outputs[numer_warstwy+1]
                    wejscie = outputs[numer_warstwy]
                    if numer_warstwy >= len(self._Network)-1:
                        gradient = warstwa.backward_sgd(y_pred=predykcja, point=wejscie, y_origin= y_train[i])
                    else:
                        gradient = warstwa.backward_sgd(y_pred=predykcja, point=wejscie,
                                                                weights_forward=self._Network[numer_warstwy + 1].wagi,
                                                                gradient2=gradient*1.5)



            loss =error/(len(y_train))
            self.train_loss.append(loss)
            train_acc,_ =self.perceptron(x_train,y_train)
            valid_acc,valid_loss = self.perceptron(x_validate,y_validate)
            self.train_accuracy.append(train_acc)
            self.valid_loss.append(valid_loss)
            self.valid_accuracy.append(valid_acc)
            try:
                if (train_acc / self.train_accuracy[-3]) >= 1.01 and train_acc >= 0.56 and valid_acc > 0.55 and help <= 10:
                    print("zmniejszenie wag", j)
                    self.alpha = self.alpha * 0.5
                    help += 1
                    for layer in self._Network:
                        layer.alfa = self.alpha

                if j % 2 == 0:
                    czy_oba_wieksze_od80 = train_acc > 0.85 and valid_acc > 0.85
                    czy_modelowi_nie_spada_jakosc = self.train_accuracy[-2] > train_acc and self.valid_accuracy[
                        -2] > valid_acc
                    czy_modelowi_nie_rosnie_blod = self.train_loss[-2] - self.train_loss[-1] <= 0.003 and self.valid_loss[
                        -2] - self.valid_loss[-1] <= 0.003
                    if (czy_oba_wieksze_od80 and (czy_modelowi_nie_spada_jakosc and czy_modelowi_nie_rosnie_blod)) or (train_acc >= 0.98 and valid_acc >= 0.90) or (valid_acc >= 0.98 and train_acc >= 0.90):
                        break
                if j >= 50 and j % 50 == 0:
                    loss_greater_than01 = self.train_loss[-1] >= 0.1 and self.valid_loss[-1] >= 0.1
                    accuracys_loss_than = train_acc < 0.60 or valid_acc < 0.60
                    if loss_greater_than01 or accuracys_loss_than:
                        print("zmiana parametrów", j)
                        for layer in self._Network:
                            layer.start(0.005)

            except Exception as e:
                print("błąd ", e)

    def train_mini_batch(self,data,batch_size):
        """

        :param data:  full data with x and y in pd.DataFrame
        :param batch_size: int szie of one batch of data input for iteration before update
        : use  mini batch backpropagation that requires optimizer  set to 'mini-batch' or 'batch'
        :return:
        """

        X = data.iloc[:, :-1].values
        Y = data.iloc[:, -1].values

        from Data.SPLIT_test_valid_train import SplitData
        splitter = SplitData()
        groups_for_model = splitter.split_in_groups(X, 6, 0.6)
        groups = splitter.get_in_order(groups_for_model)

        help = 0
        for j in range(self.epoki+1):
            train_idx, valid_idx = random.sample(groups, 2)
            x_train = X[groups_for_model[train_idx][0]]  # 0.6
            x_valid = X[groups_for_model[valid_idx][1]]  # 0.4
            y_train = Y[groups_for_model[train_idx][0]].reshape(-1, )
            y_valid = Y[groups_for_model[valid_idx][1]].reshape(-1, )


            data_train = SplitData.merge(x_train, y_train)
            batch_x, batch_y = NNetwork.split_on_batches(data_train, batch_size)


            # valid data
            #batche
            for batch_x_,batch_y_ in zip(batch_x,batch_y):
                error = 0
                #pkt w każdym batchu
                for point,y_actual  in zip(batch_x_, batch_y_):
                    outputs = []
                    output = point
                    outputs.append(output)
                    for layer in self._Network:
                        output = layer.train_forward(output)
                        outputs.append(output)
                    #print(output,y_actual,end=" ")


                    error += (y_actual - outputs[-1][0]) ** 2

                    # batch wymaga zęby gradient był wyliczany ale nie updatowany
                    for numer_warstwy in reversed(range(len(self._Network))):
                        warstwa = self._Network[numer_warstwy]
                        predykcja = outputs[numer_warstwy + 1]
                        wejscie = outputs[numer_warstwy]
                        if numer_warstwy >= len(self._Network) - 1:
                            pochodna_wyjscia = predykcja-y_actual
                            gradient = warstwa.backward_batches(y_pred=predykcja, point=wejscie, pochodna_wyjscia=pochodna_wyjscia)
                        else:
                            gradient = warstwa.backward_batches(y_pred=predykcja, point=wejscie,
                                                                    weights_forward=self._Network[
                                                                        numer_warstwy + 1].wagi,
                                                                    gradient2=gradient )
                for_average =len(batch_x_)

                for layer in self._Network:
                    layer.backward_update_params(for_average=for_average)

            loss = error / (len(batch_y_))
            self.train_loss.append(loss)
            train_acc, _ = self.perceptron(batch_x_, batch_y_)
            valid_acc, valid_loss = self.perceptron(x_valid,y_valid)
            self.train_accuracy.append(train_acc)
            self.valid_loss.append(valid_loss)
            self.valid_accuracy.append(valid_acc)

            try:
                if (train_acc / self.train_accuracy[-3]) >= 1.02 and train_acc >= 0.67 and valid_acc > 0.76 and help<=14:
                    print("zmniejszenie wag",j)
                    self.alpha = self.alpha *0.5
                    help+=1
                    for layer in self._Network:
                        layer.alfa = self.alpha
                if j%2==0:
                    czy_oba_wieksze_od80 = train_acc>0.85 and valid_acc>0.85
                    czy_modelowi_nie_spada_jakosc =self.train_accuracy[-2]>train_acc and self.valid_accuracy[-2]>valid_acc
                    czy_modelowi_nie_rosnie_blod = self.train_loss[-2]-self.train_loss[-1]<=0.005 and self.valid_loss[-2]-self.valid_loss[-1]<=0.005
                    przynajmniej_jeden_spada = self.train_loss[-2]>self.train_loss[-1] and self.valid_loss[-2]>self.valid_loss[-1]
                    if  (czy_oba_wieksze_od80 and(czy_modelowi_nie_spada_jakosc and czy_modelowi_nie_rosnie_blod )) or ((train_acc>=0.99 and valid_acc>=0.97) or (valid_acc>=0.99 and train_acc>=0.97) and  not przynajmniej_jeden_spada ) :
                        break

            except Exception as e:
                print("błąd ", e)



    def perceptron(self, x_test, y_test):
        """

        :param x_test: pd.DataFrame
        :param y_test: pd.DataFrame
        :return: accuracy,MSE error
        it is used only for validation purposes to check model stats and save informations of training porcess
        """
        predictions = []
        error = 0
        for point, y_point in zip(x_test, y_test):
            output = point

            for layer in self._Network:
                output = layer.train_forward(output)

            error += (y_point - output) ** 2
            #print(output[0], y_point)
            pred = 1 if output[0]>=0.50 else 0
            predictions.append(pred)

        strata = (error / len(y_test))[0]
        #print(predictions)
        return (sum([1 if y_pred == y_origin else 0 for y_pred, y_origin in zip(predictions, y_test)]) / len(y_test)),strata
    @log_execution_time
    def pred(self,point):
        """
        :param point:  one point of data, array with shape (n,)
        :return:  0 if output is <=51 else 1 prediction of 2 classes
        """
        point  =point
        output = point
        for layer in self._Network:
            output = layer.train_forward(output)
        return output
    @classmethod
    def confusion_matrix(cls,y_pred:list,y_origin:list):
        """

        :param y_pred:  list of proedicted points
        :param y_origin:  list of orginal points
        :return: matrix of         false   positive
                            true     x        x
                           negative  x        x
        """
        assert isinstance(y_pred,(list,set)) and isinstance(y_origin,(list,set))," print(obiekty przekazane do macierzy muszą być typu list() )"
        from numpy import zeros
        classes = set(y_pred+y_origin)
        size = classes.__len__()
        class_to_index = {label:i for i,label in enumerate(classes)}
        conf_matrix = zeros((size,size),dtype=int)
        for actual, predicted in zip(y_origin, y_pred):
            i = class_to_index[actual]
            j = class_to_index[predicted]
            conf_matrix[i][j] += 1
        return pd.DataFrame(conf_matrix,columns = list(classes))



    def after(self,show=False):
        """
        :param show: show also weights and biases
        :return:  show all layers all parameters
        """
        for layer in self._Network:
            print(f"layer : {layer.return_params(show=show)}")
    @staticmethod
    def split_on_batches(data, size):
        #data_tasowane = data.sample(frac=1).reset_index(drop=True)
        shuffled_data = SplitData.tasowanie(data, f=True)

        ilosc = shuffled_data.shape[0]

        podzial = ilosc // size

        start, stop = 0, podzial
        batch_x, batch_y = [], []
        for i in range(size - 1):
            batch_x.append(shuffled_data.iloc[start:stop, :-1].values)
            batch_y.append(shuffled_data.iloc[start:stop, -1])
            start += podzial
            stop += podzial

        batch_x.append(shuffled_data.iloc[start:, :-1].values)
        batch_y.append(shuffled_data.iloc[start:, -1])

        return batch_x, batch_y
    def write_model(self,path=r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\model.json"):
        """
        save model to passed location of data
        :param path:
        :return:
        """

        weights = [ layer.wagi.tolist()  for  layer in self._Network]
        biases = [layer.bias.tolist()  for  layer in self._Network]
        activations = [layer.activation_layer  for  layer in self._Network]
        Beta = self._Network[1].Beta1
        Beta2 = self._Network[1].Beta2
        v_weights =[np.zeros_like(layer.v_weights).tolist() for layer in self._Network ]
        v_biases = [np.zeros_like(layer.v_biases).tolist() for layer in self._Network ]
        epsilion = self._Network[1].epsilion
        # weights_exponential_d = [np.zeros_like(layer.weights_exponential_d).tolist() for layer in self._Network ]
        # biases_exponential_d =[np.zeros_like(layer.biases_exponential_d).tolist() for layer in self._Network ]
        # m_weights = [np.zeros_like(layer.m_weights).tolist() for layer in self._Network ]
        # m_biases =[np.zeros_like(layer.m_biases).tolist() for layer in self._Network ]



        #Prepare the data dictionary for saving
        model_data = {
            "weights"  : weights,
            "biases"   : biases,
            "activations": activations,
            "optimizer":self.optimizer,
            "gradients":self.gradients,
            "v_weights":v_weights,
            "v_biases" :v_biases,
            # "weights_exponential_d":weights_exponential_d,
            # "biases_exponential_d":biases_exponential_d,
            # "m_weights": m_weights ,
            # "m_biases":m_biases
        }

        # Save to JSON file
        with open(path, "w") as model_file:
            json.dump(model_data, model_file,indent=1)
            print("model saved as:",path )

    @classmethod
    def create_instance(cls,path=r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\model.json",alfa=0.1):
        with open(path, "r") as model_read:
            data = json.load(model_read)
        instance = cls(optimizer=data.get("optimizer"),gradients=data.get("gradients"),epoki=100)
        instance._Network=[]
        instance.alpha=0.05
        optimizer = data.get("optimizer")
        gradients = data.get("gradients")
        # Wyczyszczenie listy warstw jeśli istnieje
        # doać Beta rmsprop epsilion i zmienne przychowywujące biasy i
        for w, b, act,v_w,v_b in zip(
                data["weights"],
                data["biases"],
                data["activations"],
                data["v_weights"],
                data["v_biases"]
        ):
            wyjscia_ilosc,len_data =np.array(w,dtype=np.float64).shape
            instance.add_layer(inputs=len_data,outputs=wyjscia_ilosc,activation_layer_name=act)
            instance._Network[-1].wagi = np.array(w, dtype=np.float64)
            instance._Network[-1].bias = np.array(b, dtype=np.float64)
            # instance._Network[-1].weights_exponential_d = np.array(w_exp_d, dtype=np.float64)
            # instance._Network[-1].biases_exponential_d = np.array(b_exp_d, dtype=np.float64)
            instance._Network[-1].v_weights = np.array(v_w, dtype=np.float64)
            instance._Network[-1].v_biases = np.array(v_b, dtype=np.float64)
            # instance._Network[-1].m_weights = np.array(m_w, dtype=np.float64)
            # instance._Network[-1].m_biases = np.array(m_b, dtype=np.float64)
        print("Model loaded from:", path)
        return instance








