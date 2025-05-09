from NeuralNetwork.Network_single_class1 import NNetwork
def training(data):
    from Data.SPLIT_test_valid_train import SplitData
    Split = SplitData()
    # ustawiamy % danych na zbiór
    Split.set(train=0.6, valid=0.2, test=0.2)
    #podział
    x_train,y_train,x_valid,y_valid,x_test,y_test =Split.split_data(data)


    network = NNetwork(epoki=100,alpha=0.35,optimizer="momentum",gradients="batch") #optimizer="momentum",gradients="batch"
    network.add_layer(31,12,"relu")
    network.add_layer(12,12,"relu")
    network.add_layer(12,1,"sigmoid")

    network.train_mini_batch(x_train,y_train,x_valid,y_valid)
    net_loss = network.loss
    net_acc = network.train_accuracy
    valid_loss = network.valid_loss
    valid_accuracy=network.valid_accuracy
    test_acc, test_loss = network.perceptron(x_test, y_test)
    print("train loss:  ",net_loss[-1],    " train acc: ", net_acc[-1],)
    print("valid loss:  ",valid_loss[-1], "  valid acc: ",valid_accuracy[-1] )
    print("test loss:   ", test_loss,           "  test acc   ",test_acc )
    from NeuralNetwork.Show_results import show_training_process
    show_training_process(network.train_accuracy,network.loss,network.valid_accuracy,network.valid_loss,test_acc,test_loss)

    print("\n", network.after())

    data_x = data.iloc[:,:-1].values
    y = data.loc[:,"y"].values.tolist()
    # model_instance = NNetwork.create_instance()
    # skutecznosc, strata = model_instance.perceptron(data_x, y)
    skutecznosc,strata = network.perceptron(data_x,y)
    print("test accuracy:   ", skutecznosc, "  test loss  ", strata)




