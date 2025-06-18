def show_training_process(train_acc: list, train_loss: list, valid_acc: list, valid_loss: list, test_acc: float,
                          test_loss: float, index:int  = 1):
    """
    Visualizes the training, validation, and test performance of a neural network over training epochs.

    This function plots training and validation loss and accuracy across epochs using a dual y-axis format.

    :param train_acc: List of training accuracy values .
    :param train_loss: List of training loss values .
    :param valid_acc: List of validation accuracy values .
    :param valid_loss: List of validation loss values .
    :param test_acc: Final test set accuracy (a single float value).
    :param test_loss: Final test set loss (a single float value).
    :raises AssertionError: If the lengths of input lists for training and validation metrics are not equal.
    """
    assert(len(train_loss)==len(train_acc)==len(valid_acc)==len(valid_loss)),f"dane powinny zaweirać taki sam rozmiar{len(train_acc)}=={len(train_acc)}=={len(valid_acc)}=={len(valid_loss)}"
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(111)
    ax1.set_ylim([0,1.1])

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss/Accuracy Network")
    maxi = len(train_acc) - 1
    #plot lines
    ax1.plot(train_loss, label="train Loss", color=(0.5, 0.3, 0.1, 0.9))  # red green blue alpha
    ax1.plot(train_acc, label="train Accuracy", color=(0.6, 0.4, 0.1, 0.4))
    # Scatter test points
    ax1.scatter(maxi, test_acc, s=80,
            color=(0.5, 0.1, 0.7, 0.9), edgecolors='pink', label="acc", linewidths=0.8, zorder=5)
    ax1.scatter(maxi, test_loss, s=80,
            color=(0.1, 0.2, 0.99, 0.9), edgecolors='pink', label="loss", linewidths=0.8, zorder=5)
    # ✅ Add annotations (text above points)
    ax1.annotate(f"{test_acc:.2f}", (maxi, test_acc), textcoords="offset points", xytext=(0, 8), ha='center',fontsize=8)
    ax1.annotate(f"{test_loss:.2f}", (maxi, test_loss), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)
    ax1.grid(True)
    # create twin axis for validation data
    ax2 = ax1.twinx()
    ax2.set_ylim([0, 1.1])

    ax2.plot(valid_loss, label="valid Loss", color=(0.2, 0.7, 0.4, 0.9))
    ax2.plot(valid_acc, label="valid Accuracy", color=(0.1, 0.8, 0.4, 0.5))
    ax2.grid(True)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.title(f"Loss and Accuracy over Epochs {index}")
    plt.show()
