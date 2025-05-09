# Tu Będzie zbudowana sieć neuronowa która edzie miałą warstwy złożone z One_Layer
import json
from NeuralNetwork.One_Layer import LayerFunctions

class NNetwork():
    __slots__ = ["epoki","alpha","LineOne","LineTwo","LineThree","loss","train_accuracy","valid_accuracy", "valid_loss"]
    def __init__(self,w=None,epoki=None,alpha=0.02,batche = None):
        self.epoki = epoki
        self.alpha = alpha
        if epoki is None:
            self.epoki = 2
        #Przechowanie warstw jako lista oraz utrzymanie struktury sieci jako listy

        self.LineOne = LayerFunctions(len_data=31,wyjscie_ilosc=12,activation_layer="relu")
        self.LineTwo = LayerFunctions(len_data=12,wyjscie_ilosc=12,activation_layer="elu")
        self.LineThree = LayerFunctions(len_data=12,wyjscie_ilosc=1,activation_layer="sigmoid")
        self.LineOne.start(self.alpha)
        self.LineTwo.start(self.alpha)
        self.LineThree.start(self.alpha)




        # self.LineOne.bias = np.array([0.5, 0.5, 0.5])
        # self.LineTwo.bias = np.array([1., 1., 1.])
        # self.LineThree.bias = np.array([1.])
        #
        # self.LineOne.wagi= np.array([[0.6,0.5,0.5], [0.5,0.5,0.5], [0.5,0.4,0.5]])
        # self.LineTwo.wagi = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        # self.LineThree.wagi = np.array([[0.5, 0.4, 0.4]])

        self.loss = []
        self.valid_loss= []
        self.train_accuracy = []
        self.valid_accuracy = []

    def train(self,x_train,y_train,x_validate,y_validate):
        for j in range(1, self.epoki + 1):
            error = 0
            for i, point in enumerate(x_train):
                ## forward pass
                output = self.LineOne.train_forward(point)
                output2 = self.LineTwo.train_forward(output)
                output3 = self.LineThree.train_forward(output2)

                # cross entropy error
                error += 0.5*(y_train[i]-output3[0])**2

                ## backward pass
                backward_step_third_line = self.LineThree.backward_sgd(y_pred=output3, point=output2,
                                                                               y_origin=y_train[i])

                backward_step_second_line = self.LineTwo.backward_sgd(y_pred=output2, point=output,
                                                                              weights_forward=self.LineThree.wagi,
                                                                              gradient2=backward_step_third_line)

                backward_step_first_line= self.LineOne.backward_sgd(y_pred=output, point=point,
                                                                            weights_forward=self.LineTwo.wagi,
                                                                            gradient2=backward_step_second_line * 2)

            loss =error/(len(y_train))
            self.loss.append(loss)
            train_acc,_ =self.perceptron(x_train,y_train)
            valid_acc,valid_loss = self.perceptron(x_validate,y_validate)
            self.train_accuracy.append(train_acc)
            self.valid_loss.append(valid_loss)
            self.valid_accuracy.append(valid_acc)

            if j%5==0 and (((train_acc/self.train_accuracy[-2])>=1.02 and train_acc>= 0.7) or valid_acc>0.75):
                print("zmniejszenie wag")
                self.alpha = self.alpha*0.5
                self.LineOne.alfa=self.alpha
                self.LineTwo.alfa=self.alpha
                self.LineThree.alfa=self.alpha
                if all(round(self.valid_loss[-1],5) >= round(x,5) for x in self.valid_loss[-4:-1]) and self.valid_loss[-1]>self.loss[-1]:
                    print("przerwanie możliwe przeuczenie")
                    break
                if j%10==0 and ((train_acc==self.train_accuracy[-5]) and
                         (sum(self.valid_accuracy[-5:])/5==self.valid_accuracy[-1])):
                    print("przerwanie")
                    break



            ## warunek zlaerzny od ostatniej wartości  accuracy i np grarancji że błąd nie rośnie


    def perceptron(self, x_test, y_test):
        predictions = []
        error = 0
        for  point, y_point in zip(x_test, y_test):
            output = self.LineOne.train_forward(point)
            output2 = self.LineTwo.train_forward(output)
            output3 = self.LineThree.train_forward(output2)
            error += 0.5*(y_point-output3)**2

            pred = round(output3[0])
            predictions.append(pred)
        strata= (error / len(y_test))[0]
        return (sum([1 if y_pred == y_origin else 0 for y_pred, y_origin in zip(predictions, y_test)]) / len(y_test)),strata

    def pred(self,point):
        point  =point.T
        output = self.LineOne.train_forward(point)
        output2 = self.LineTwo.train_forward(output)
        output3 = self.LineThree.train_forward(output2.T)
        return output3


    def after(self):
        print(self.LineOne.wagi)
        print(self.LineOne.bias)
        print(self.LineTwo.wagi)
        print(self.LineTwo.bias)
        print(self.LineThree.wagi)
        print(self.LineThree.bias)

    def write_model(self):
        file = r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\model.json"
        weights = [self.LineOne.wagi.tolist(), self.LineTwo.wagi.tolist(), self.LineThree.wagi.tolist()]
        biases = [self.LineOne.bias.tolist(), self.LineTwo.bias.tolist(), self.LineThree.bias.tolist()]
        activations = [self.LineOne.activation_layer,self.LineTwo.activation_layer,self.LineThree.activation_layer]
        # Prepare the data dictionary for saving
        model_data = {
            "weights": weights,
            "biases": biases,
            "activations": activations
        }

        # Save to JSON file
        with open(file, "w") as model_file:
            json.dump(model_data, model_file,indent=2)
            print("model saved")



    @classmethod
    def create_instance(cls):
        from numpy import array
        # Define file path
        file = r"C:\Program Files\Pulpit\Data_science\Gui_szkolenie_4\TrainData\model.json"

        # Read model data from JSON
        with open(file, "r") as model_read:
            data = json.load(model_read)

        # Create instance of the model
        instance = cls()

        # Load weights and biases for each layer
        instance.LineOne.wagi = array(data["weights"][0])
        instance.LineOne.bias = array(data["biases"][0])
        instance.LineOne.activation_layer = data["activations"][0]

        instance.LineTwo.wagi = array(data["weights"][1])
        instance.LineTwo.bias = array(data["biases"][1])
        instance.LineTwo.activation_layer = data["activations"][1]

        instance.LineThree.wagi = array(data["weights"][2])
        instance.LineThree.bias = array(data["biases"][2])
        instance.LineThree.activation_layer = data["activations"][2]

        return instance





