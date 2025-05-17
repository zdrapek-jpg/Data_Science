# Tu Będzie zbudowana sieć neuronowa która edzie miałą warstwy złożone z One_Layer
import json
from Data.SPLIT_test_valid_train import SplitData
from NeuralNetwork.One_Layer import LayerFunctions
from Data.Decorator_time_logging import log_execution_time
class NNetwork():
    __slots__ = ["epoki","alpha","_Network","loss","train_accuracy","valid_accuracy", "valid_loss","optimizer","gradients"]
    def __init__(self,w=None,epoki=None,alpha=0.02,optimizer=None,gradients=None):
        self.epoki = epoki
        self.alpha = alpha
        if epoki is None:
            self.epoki = 2
        #Przechowanie warstw jako lista oraz utrzymanie struktury sieci jako listy
        self._Network = []
        self.optimizer = optimizer
        self.gradients=gradients


        self.loss = []
        self.valid_loss= []
        self.train_accuracy = []
        self.valid_accuracy = []

    def add_layer(self,inputs,outputs,activation_layer_name):
        # instancja klasy warstwy
        instance_of_layer = LayerFunctions(len_data= inputs,wyjscie_ilosc= outputs,activation_layer=activation_layer_name,optimizer=self.optimizer,gradients=self.gradients)
        instance_of_layer.start(self.alpha)


        if len(self._Network)>=1:
            if inputs!= self._Network[-1].wyjscia_ilosc:
                raise f"{inputs}!= {self._Network[-1].wyjscia_ilosc} powinny być takie same"
        self._Network.append(instance_of_layer)

    def train_sgq(self,x_train,y_train,x_validate,y_validate):
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
                error += 0.5*(y_train[i]-outputs[-1][0])**2

                for  numer_warstwy in reversed(range(len(self._Network))):
                    warstwa = self._Network[numer_warstwy]
                    predykcja = outputs[numer_warstwy+1]
                    wejscie = outputs[numer_warstwy]
                    if numer_warstwy >= len(self._Network)-1:
                        gradient = warstwa.backward_sgd(y_pred=predykcja, point=wejscie, y_origin= y_train[i])
                    else:
                        gradient = warstwa.backward_sgd(y_pred=predykcja, point=wejscie,
                                                                weights_forward=self._Network[numer_warstwy + 1].wagi,
                                                                gradient2=gradient*1.3)



            loss =error/(len(y_train))
            self.loss.append(loss)
            train_acc,_ =self.perceptron(x_train,y_train)
            valid_acc,valid_loss = self.perceptron(x_validate,y_validate)
            self.train_accuracy.append(train_acc)
            self.valid_loss.append(valid_loss)
            self.valid_accuracy.append(valid_acc)
            try:
                if j%2==0 and (((train_acc/self.train_accuracy[-2])>=1.02 or train_acc>= 0.6) and valid_acc>0.5):
                    print("zmniejszenie wag")
                    self.alpha = self.alpha**2
                    for layer in self._Network:
                        layer.alfa = self.alpha
                    if j%5==0 and ((train_acc==self.train_accuracy[-4]) and
                            any( self.train_accuracy[i]<=self.train_accuracy[i-1] and self.valid_accuracy[i]<=self.valid_accuracy[i-1] for i in range(len(self.train_accuracy)-4,len(self.train_accuracy)-1)) or
                                    any(round(self.loss[-1],5) >= round(x,5) for x in self.loss[-4:-1])):
                        print("przerwanie")
                        break
            except Exception as e:
                print("błąd ",e)

    def train_mini_batch(self,x_train,y_train,x_valid,y_valid):
        #print(self._Network[0].return_params())
        # epoki

        data_train = SplitData.merge(x_train, y_train)


        help = 0
        for j in range(self.epoki+1):
            batch_x, batch_y = NNetwork.split_on_batches(data_train, 32)
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


                    error += 0.5 * (y_actual - outputs[-1][0]) ** 2

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
                    layer.backward_batches(y_pred=y_actual,for_average=for_average)

            loss = error / (len(batch_y_))
            self.loss.append(loss)
            train_acc, _ = self.perceptron(batch_x_, batch_y_)
            valid_acc, valid_loss = self.perceptron(x_valid,y_valid)
            self.train_accuracy.append(train_acc)
            self.valid_loss.append(valid_loss)
            self.valid_accuracy.append(valid_acc)

            try:
                if j%5==0 and  (train_acc / self.train_accuracy[-3]) >= 1.01 and train_acc >= 0.56 and valid_acc > 0.55 and help<=6:
                    print("zmniejszenie wag",j)
                    self.alpha = self.alpha *0.5
                    help+=1
                    for layer in self._Network:
                        layer.alfa = self.alpha

                if j%2==0:
                    czy_oba_wieksze_od80 = train_acc>0.85 and valid_acc>0.85
                    czy_modelowi_nie_spada_jakosc =self.train_accuracy[-2]>train_acc and self.valid_accuracy[-2]>valid_acc
                    czy_modelowi_nie_rosnie_blod = self.loss[-2]-self.loss[-1]>=0.013 and self.valid_loss[-2]-self.valid_loss[-1]>=0.013
                    if  (czy_oba_wieksze_od80 and(czy_modelowi_nie_spada_jakosc and czy_modelowi_nie_rosnie_blod )) or train_acc>=0.99 or valid_acc>=0.99 :
                        break
                if j >=50 and j%50==0:
                    loss_greater_than01 = self.loss[-1] >= 0.1 and self.valid_loss[-1] >= 0.1
                    accuracys_loss_than = train_acc < 0.60 or valid_acc < 0.60
                    if  loss_greater_than01  or accuracys_loss_than :
                        print("zmiana parametrów", j)
                        for layer in self._Network:
                            layer.start(0.5)

            except Exception as e:
                print("błąd ", e)



    def perceptron(self, x_test, y_test):
        predictions = []
        error = 0
        for point, y_point in zip(x_test, y_test):
            output = point
            for layer in self._Network:
                output = layer.train_forward(output)
            error += 0.5 * (y_point - output) ** 2
            #print(output[0], y_point)
            pred = round(output[0])
            predictions.append(pred)
        strata = (error / len(y_test))[0]
        #print(predictions)
        return (sum([1 if y_pred == y_origin else 0 for y_pred, y_origin in zip(predictions, y_test)]) / len(y_test)),strata
    @log_execution_time
    def pred(self,point):
        point  =point.T
        output = point
        for layer in self._Network:
            output = layer.train_forward(output)
        return output


    def after(self):
        for layer in self._Network[1:]:
            print(f"layer : {layer.return_params()}")
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
            print("model saved as:",path )

    @classmethod
    def create_instance(cls,path=r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\model.json"):
        import json
        from numpy import array

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








