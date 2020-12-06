import matplotlib.pyplot as plt
import numpy as np
import pickle
#
def loss_plotting(data):
    fig, axs = plt.subplots(1, len(data))
    if len(data)==3:
        plots_names = {0: 'train loss', 1: 'validation loss', 2: 'test loss'}
    if len(data)==2:
        plots_names = {0: 'validation loss', 1: 'test loss'}
    for i in range(len(data)):
        axs[i].set_xlabel("Episodes", fontsize=20, labelpad=10)
        axs[i].set_ylabel(plots_names[i], fontsize=24, labelpad=3)
        axs[i].plot(np.array(data[plots_names[i]]), 'k--', linewidth=2)
    # axs[len(data)].legend(loc='upper left', prop={'size': 18})
    plt.show()













