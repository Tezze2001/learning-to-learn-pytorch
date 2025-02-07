import os

import numpy as np
import torch
import torch.optim as optim

from impl.mnist_problem import MNISTLoss, MNISTNet
from impl.optimizer import Optimizer
from impl.utils import fit_outer_model, fit_std_optimizer, move_to_cuda
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


path_results = './results/from_original_impl'
n_layers = [1, 2, 3, 4, 5]
layer_size = 5

if not os.path.isfile('./models/meta-optimizer-mnist-0_01.pth'):
    print('Not found best model for mnist so stopped the execution')
    exit()
else:
    meta_optimizer_mnist_lr_0_01 = torch.load('./models/meta-optimizer-mnist-0_01.pth', weights_only=True)

if not os.path.isfile(f'{path_results}/results_mnist_number_layers_test.npz'):
    NORMAL_OPTS = [(optim.Adam, {}), (optim.RMSprop, {}), (optim.SGD, {}), (optim.SGD, {'nesterov': True, 'momentum': 0.9})]
    OPT_NAMES = ['ADAM', 'RMSprop', 'SGD', 'NAG']

    QUAD_LRS = [0.01, 0.01, 0.1, 0.1]
    fit_data_mnist = np.zeros((100, 100, (len(OPT_NAMES) + 1)* len(n_layers)))
    for i, ((opt, extra_kwargs), lr) in enumerate(zip(NORMAL_OPTS, QUAD_LRS)):
        for j, n in enumerate(n_layers):
            torch.manual_seed(42)
            MNISTNet_partial = partial(MNISTNet, layer_size=layer_size, n_layers=n)
            fit_data_mnist[:, :, i * len(n_layers) + j] = np.array(fit_std_optimizer(MNISTLoss, MNISTNet_partial, opt, lr=lr, **extra_kwargs))

    opt = move_to_cuda(Optimizer(preproc=True))
    opt.load_state_dict(meta_optimizer_mnist_lr_0_01)
    for j, n in enumerate(n_layers):
        torch.manual_seed(42)
        MNISTNet_partial = partial(MNISTNet, layer_size=layer_size, n_layers=n)
        fit_data_mnist[:, :, len(OPT_NAMES) * len(n_layers) + j] = np.array([fit_outer_model(opt, None, MNISTLoss, MNISTNet_partial, unroll=1, n_steps_to_optimize=100, out_mul=0.1, should_train=2) for _ in range(100)])


    np.savez(f'{path_results}/results_mnist_number_layers_test.npz', data=fit_data_mnist)
else: 
    fit_data_mnist = np.load(f'{path_results}/results_mnist_number_layers_test.npz')['data']
    print('risultati caricati da file')

def plot_performance(name_optimizers, number_neurons, fit_data, title_plot, n_epochs=100, n_optimization_steps=100):

    colors = sns.color_palette("Set2", n_colors=len(name_optimizers))

    for l, n_neuron in enumerate(number_neurons):

        df_list = []
        for i, method in enumerate(name_optimizers):
            for j in range(n_epochs):  # 100 esecuzioni
                for k in range(n_optimization_steps):  # 100 esecuzioni
                    df_list.append({'Steps': j, 'Loss': fit_data[k, j, i * len(number_neurons) + l],  'Method': method if method != 'LSTM' else 'CLSTMOpt'})

        df = pd.DataFrame(df_list)

        # Creiamo il lineplot con intervalli di confidenza automatici
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=df, x="Steps", y="Loss", hue="Method", palette=colors, errorbar=('ci', 95)) 

        # Personalizziamo la scala e gli elementi del plot
        plt.yscale('log')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title(f"Optimization performance of FFNN with {n_neuron} hidden layer with 5 neurons each one executed on mnist")
        plt.legend()
        plt.show()
    
plot_performance(['ADAM', 'RMSprop', 'SGD', 'NAG', 'CLSTMOpt'], n_layers, fit_data_mnist, "varing number of layers")