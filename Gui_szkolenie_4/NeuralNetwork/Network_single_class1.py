# Tu Będzie zbudowana sieć neuronowa która edzie miałą warstwy złożone z One_Layer
import json

import numpy as np

from NeuralNetwork.One_Layer import LayerFunctions

class NNetwork():
    __slots__ = ["epoki","alpha","_Network","loss","train_accuracy","valid_accuracy", "valid_loss","optimizer"]
    def __init__(self,w=None,epoki=None,alpha=0.02,batche = None,optimizer=None):
        self.epoki = epoki
        self.alpha = alpha
        if epoki is None:
            self.epoki = 2
        #Przechowanie warstw jako lista oraz utrzymanie struktury sieci jako listy
        self._Network = []
        self.optimizer = optimizer


        self.loss = []
        self.valid_loss= []
        self.train_accuracy = []
        self.valid_accuracy = []

    def add_layer(self,inputs,outputs,activation_layer_name):
        # instancja klasy warstwy
        instance_of_layer = LayerFunctions(len_data= inputs,wyjscie_ilosc= outputs,activation_layer=activation_layer_name)
        instance_of_layer.start(self.alpha)

        if len(self._Network)>=1:
            if inputs!= self._Network[-1].wyjscia_ilosc:
                raise f"{inputs}!= {self._Network[-1].wyjscia_ilosc} powinny być takie same"
        self._Network.append(instance_of_layer)




    def train(self,x_train,y_train,x_validate,y_validate):
        for j in range(1, self.epoki + 1):
            error = 0
            for i, point in enumerate(x_train):
                ## forward pass
                outputs =[point]
                output = point
                for layer in self._Network:
                    output = layer.train_forward(output)
                    outputs.append(output)
                print(output)



                # cross entropy error
                error += 0.5*(y_train[i]-outputs[-1])**2
                gradients = []
                gradient = None
                for  numer_warstwy in range(len(self._Network)):
                    warstwa = self._Network[numer_warstwy]
                    predykcja = outputs[numer_warstwy]
                    wejscie = outputs[numer_warstwy-1]
                    if numer_warstwy >= len(self._Network)-1:
                        gradient =warstwa.backward_sgd(y_pred=predykcja,point=wejscie,y_origin = y_train[i])
                    else:
                        gradient =warstwa.backward_sgd(y_pred=predykcja, point=wejscie, y_origin=y_train[i],weights_forward=self._Network[numer_warstwy+1].wagi,gradient2=gradient)
                    gradients.append(gradient)


            loss =error/(len(y_train))
            self.loss.append(loss)
            train_acc,_ =self.perceptron(x_train,y_train)
            valid_acc,valid_loss = self.perceptron(x_validate,y_validate)
            self.train_accuracy.append(train_acc)
            self.valid_loss.append(valid_loss)
            self.valid_accuracy.append(valid_acc)

            if j%5==0 and (((train_acc/self.train_accuracy[-2])>=1.02 or train_acc>= 0.6) and valid_acc>0.65):
                print("zmniejszenie wag")
                self.alpha = self.alpha*0.5
                for layer in self._Network:
                    layer.alfa = self.alpha
                if j%10==0 and (((train_acc==self.train_accuracy[-5]) and
                         (sum(self.valid_accuracy[-5:])/5==self.valid_accuracy[-1])) or
                                any(round(self.loss[-1],5) >= round(x,5) for x in self.loss[-5:-1])):
                    print("przerwanie")
                    break



            ## warunek zlaerzny od ostatniej wartości  accuracy i np grarancji że błąd nie rośnie


    def perceptron(self, x_test, y_test):
        predictions = []
        error = 0
        for i, (point, y_point) in enumerate(zip(x_test, y_test)):
            for layer in self._Network[1:]:
                output = layer.train_forward(point)
            print(point)

            pred = round(output[0])
            predictions.append(pred)
        strata= (error / len(y_test))[0]
        return (sum([1 if y_pred == y_origin else 0 for y_pred, y_origin in zip(predictions, y_test)]) / len(y_test)),strata

    def pred(self,point):
        point  =point.T
        for layer in self._Network[1:]:
            output = layer.train_forward(point)
        print(point)
        return output


    def after(self):
        for layer in self._Network[1:]:
            print(f"layer : {layer.return_params()}")

    def split_on_batches(self, data, size):
        data_tasowane = data.sample(frac=1).reset_index(drop=True)

        ilosc = data_tasowane.shape[0]

        podzial = ilosc // size

        start, stop = 0, podzial
        batch_x, batch_y = [], []
        for i in range(size - 1):
            batch_x.append(data_tasowane.iloc[start:stop, :-1].values)
            batch_y.append(data_tasowane.iloc[start:stop, -1])
            start += podzial
            stop += podzial

        batch_x.append(data_tasowane.iloc[start:, :-1].values)
        batch_y.append(data_tasowane.iloc[start:, -1])

        return batch_x, batch_y
    def write_model(self,path=r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\model.json"):

        weights = [ layer.wagi.tolist()  for  layer in self._Network]
        biases = [layer.bias.tolist()  for  layer in self._Network]
        activations = [layer.activation_layer  for  layer in self._Network]
        # Prepare the data dictionary for saving
        model_data = {
            "weights": weights,
            "biases": biases,
            "activations": activations
        }

        # Save to JSON file
        with open(path, "w") as model_file:
            json.dump(model_data, model_file,indent=2)
            print("model saved")

    @classmethod
    def create_instance(cls,path=r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\model.json"):
        import json
        from numpy import array

        path= r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\model.json"

        with open(path, "r") as model_read:
            data = json.load(model_read)

        instance = cls()

        # Wyczyszczenie listy warstw jeśli istnieje
        instance._Network = []

        for w, b, act in zip(data["weights"], data["biases"], data["activations"]):
            new_layer = LayerFunctions(len_data=len(w), wyjscie_ilosc=len(b), activation_layer=act)
            new_layer.wagi = array(w)
            new_layer.bias = array(b)
            instance._Network.append(new_layer)

        print("Model loaded from:", path)
        return instance








