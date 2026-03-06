import torch
import numpy as np 
import matplotlib.pyplot as plt

def plot_loss_and_validation_loss(train_loss_list, test_loss_list) -> None:
    list_loss_train = []
    list_loss_mean = []
    for tensor_list_epoch in train_loss_list:
        list_epoch = []
        for tensor_loss in tensor_list_epoch:
            if type(tensor_loss) is torch.Tensor:
                list_loss_train.append(tensor_loss.item())
                list_epoch.append(tensor_loss.item())
            else:
                list_loss_train.append(tensor_loss)
                list_epoch.append(tensor_loss)
        list_loss_mean.append(np.array(list_epoch).mean())

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))

    ax.plot(range(10*107), list_loss_train, color = 'green', label='Trainings loss')
    ax.plot(range(0, 10*107, 107), list_loss_mean, color = 'blue', label='Mean trainings loss per epoch')
    ax.plot(range(107, 10*107+1, 107), test_loss_list, color = 'orange', label='Test loss')
    ax.legend()
    ax.set_title('Trainings and test loss')
