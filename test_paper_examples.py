import glob
from turtle import pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import multiprocessing
import os.path
import csv
import copy
from impl.mnist_problem import MNISTLoss, MNISTNet
from impl.optimizer import Optimizer
from impl.quadratic_problem import QuadOptimizee, QuadraticLoss
from impl.utils import fit_outer_model, fit_std_optimizer, grid_search_lr_meta_opt, grid_search_lr_std_opt, move_to_cuda, train_outer_model
import joblib
from torchvision import datasets
import torchvision
import seaborn as sns
import pandas as pd

sns.set_style("white")

path_results= './results/from_original_impl'
trainign = './results/from_original_impl'

grid_search_lr_meta_opt(f"{path_results}/grid_search_lr_meta_optimizer_quadratic.csv", 
                        QuadraticLoss, QuadOptimizee, lr_values = [1.0, 0.1, 0.01, 0.001, 0.0001], preproc=False)

grid_search_lr_meta_opt(f"{path_results}/grid_search_lr_meta_optimizer_mnist.csv", 
                        MNISTLoss, MNISTNet, lr_values = [1.0, 0.1, 0.01, 0.001, 0.0001], preproc=True)

grid_search_lr_std_opt(f"{path_results}/grid_search_lr_std_opt_quadratic.csv",
                       QuadraticLoss, QuadOptimizee)

grid_search_lr_std_opt(f"{path_results}/grid_search_lr_std_opt_mnist.csv",
                       MNISTLoss, MNISTNet)

if not os.path.isfile(f"{path_results}/training_meta_optimizer_quadratic.csv"):
    with open(f"{path_results}/training_meta_optimizer_quadratic.csv", mode="w", newline="") as file:
        writer = csv.writer(file, delimiter='|')
        writer.writerow(["model", "best_val", "train_loss_trajectory", "val_loss_trajectory"])
        # Grid search 
        best_validation_loss, best_model, train_loss_trajectory, val_loss_trajectory = train_outer_model(QuadraticLoss, QuadOptimizee, n_epochs=100, lr=0.01, out_mul=1, preproc=False)
        print('Best val loss:', best_validation_loss)
        torch.save(best_model, './models/meta-optimizer-quadratic-0_01.pth')
        writer.writerow([f"meta-opt-0.01", best_validation_loss, train_loss_trajectory, val_loss_trajectory])

        best_validation_loss, best_model, train_loss_trajectory, val_loss_trajectory = train_outer_model(QuadraticLoss, QuadOptimizee, n_epochs=100, lr=0.1, out_mul=1, preproc=False)
        print('Best val loss:', best_validation_loss)
        torch.save(best_model, './models/meta-optimizer-quadratic-0_1.pth')
        writer.writerow([f"meta-opt-0.1", best_validation_loss, train_loss_trajectory, val_loss_trajectory])
    print("End training on quadratic")
else:
    print('Log file already exists, trianing is not performed')
    

meta_optimizer_quadratic_lr_0_01 = torch.load('./models/meta-optimizer-quadratic-0_01.pth', weights_only=True)
meta_optimizer_quadratic_lr_0_1 = torch.load('./models/meta-optimizer-quadratic-0_1.pth', weights_only=True)

if not os.path.isfile(f'{path_results}/results_quadratic_test.npz'):
    NORMAL_OPTS = [(optim.Adam, {}), (optim.RMSprop, {}), (optim.SGD, {}), (optim.SGD, {'nesterov': True, 'momentum': 0.9})]
    OPT_NAMES = ['ADAM', 'RMSprop', 'SGD', 'NAG']

    QUAD_LRS = [0.1, 0.01, 0.01, 0.01]
    fit_data_quadratic = np.zeros((100, 100, len(OPT_NAMES) + 2))
    for i, ((opt, extra_kwargs), lr) in enumerate(zip(NORMAL_OPTS, QUAD_LRS)):
        torch.manual_seed(42)
        fit_data_quadratic[:, :, i] = np.array(fit_std_optimizer(QuadraticLoss, QuadOptimizee, opt, lr=lr, **extra_kwargs))

    opt = move_to_cuda(Optimizer())
    torch.manual_seed(42)
    opt.load_state_dict(meta_optimizer_quadratic_lr_0_01)
    fit_data_quadratic[:, :, len(OPT_NAMES)] = np.array([fit_outer_model(opt, None, QuadraticLoss, QuadOptimizee, unroll=1, n_steps_to_optimize=100, should_train=2) for _ in range(100)])

    torch.manual_seed(42)
    opt.load_state_dict(meta_optimizer_quadratic_lr_0_1)
    fit_data_quadratic[:, :, len(OPT_NAMES) + 1] = np.array([fit_outer_model(opt, None, QuadraticLoss, QuadOptimizee, unroll=1, n_steps_to_optimize=100, should_train=2) for _ in range(100)])

    np.savez(f'{path_results}/results_quadratic_test.npz', data=fit_data_quadratic)
else: 
    fit_data_quadratic = np.load(f'{path_results}/results_quadratic_test.npz')['data']
    print('risultati caricati da file')


meta_optimizer_mnist_lr_0_0001 = None
meta_optimizer_mnist_lr_0_01 = None
if not os.path.isfile(f"{path_results}/training_meta_optimizer_mnist.csv"):
    with open(f"{path_results}/training_meta_optimizer_mnist.csv", mode="w", newline="") as file:
        writer = csv.writer(file, delimiter='|')
        writer.writerow(["model", "best_val", "train_loss_trajectory", "val_loss_trajectory"])
        # Grid search 
        best_validation_loss, meta_optimizer_mnist_lr_0_0001, train_loss_trajectory, val_loss_trajectory = train_outer_model(MNISTLoss, MNISTNet, n_epochs=100, lr=0.0001, out_mul=.1, preproc=True)
        print('Best val loss:', best_validation_loss)
        torch.save(meta_optimizer_mnist_lr_0_0001, './models/meta-optimizer-mnist-0_0001.pth')
        writer.writerow([f"meta-opt-0.1", best_validation_loss, train_loss_trajectory, val_loss_trajectory])

        best_validation_loss, meta_optimizer_mnist_lr_0_01, train_loss_trajectory, val_loss_trajectory = train_outer_model(MNISTLoss, MNISTNet, n_epochs=100, lr=0.01, out_mul=.1, preproc=True)
        print('Best val loss:', best_validation_loss)
        torch.save(meta_optimizer_mnist_lr_0_01, './models/meta-optimizer-mnist-0_01.pth')
        writer.writerow([f"meta-opt-0.01", best_validation_loss, train_loss_trajectory, val_loss_trajectory])
    print("End training on mnist")
else:
    print('Log file already exists, trianing is not performed')
if meta_optimizer_mnist_lr_0_0001 == None:
    meta_optimizer_mnist_lr_0_0001 = torch.load('./models/meta-optimizer-mnist-0_0001.pth', weights_only=True)
if meta_optimizer_mnist_lr_0_01 == None:
    meta_optimizer_mnist_lr_0_01 = torch.load('./models/meta-optimizer-mnist-0_01.pth', weights_only=True)




if not os.path.isfile(f'{path_results}/results_mnist_test.npz'):
    NORMAL_OPTS = [(optim.Adam, {}), (optim.RMSprop, {}), (optim.SGD, {}), (optim.SGD, {'nesterov': True, 'momentum': 0.9})]
    OPT_NAMES = ['ADAM', 'RMSprop', 'SGD', 'NAG']

    QUAD_LRS = [0.01, 0.01, 0.1, 0.1]
    fit_data_mnist = np.zeros((100, 100, len(OPT_NAMES) + 2))
    for i, ((opt, extra_kwargs), lr) in enumerate(zip(NORMAL_OPTS, QUAD_LRS)):
        torch.manual_seed(42)
        fit_data_mnist[:, :, i] = np.array(fit_std_optimizer(MNISTLoss, MNISTNet, opt, lr=lr, **extra_kwargs))

    opt = move_to_cuda(Optimizer(preproc=True))
    torch.manual_seed(42)
    opt.load_state_dict(meta_optimizer_mnist_lr_0_0001)
    fit_data_mnist[:, :, len(OPT_NAMES)] = np.array([fit_outer_model(opt, None, MNISTLoss, MNISTNet, unroll=1, n_steps_to_optimize=100, out_mul=0.1, should_train=2) for _ in range(100)])

    torch.manual_seed(42)
    opt.load_state_dict(meta_optimizer_mnist_lr_0_01)
    fit_data_mnist[:, :, len(OPT_NAMES) + 1] = np.array([fit_outer_model(opt, None, MNISTLoss, MNISTNet, unroll=1, n_steps_to_optimize=100, out_mul=0.1, should_train=2) for _ in range(100)])

    np.savez(f'{path_results}/results_mnist_test.npz', data=fit_data_mnist)
else: 
    fit_data_mnist = np.load(f'{path_results}/results_mnist_test.npz')['data']
    print('risultati caricati da file')


def plot_performance(name_optimizers, fit_data, title_plot,n_epochs=100, n_optimization_steps=100):

    colors = sns.color_palette("Set2", n_colors=len(name_optimizers))


    df_list = []
    for i, method in enumerate(name_optimizers):
        for j in range(n_epochs):  # 100 esecuzioni
            for k in range(n_optimization_steps):  # 100 esecuzioni
                df_list.append({'Steps': j, 'Loss': fit_data[k, j, i], 'Method': method})

    df = pd.DataFrame(df_list)

    # Creiamo il lineplot con intervalli di confidenza automatici
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="Steps", y="Loss", hue="Method", palette=colors, errorbar=('ci', 95)) 

    # Personalizziamo la scala e gli elementi del plot
    plt.yscale('log')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title(title_plot)
    plt.legend()
    plt.show()

#mnist
plot_performance( ['ADAM', 'RMSprop', 'SGD', 'NAG', 'CLSTMOpt lr=0.01', 'CLSTMOpt lr=0.1'], fit_data_quadratic, "Performance on quadratic problems")
#plot_performance( ['ADAM', 'RMSprop', 'SGD', 'NAG', 'CLSTMOpt lr=0.0001', 'CLSTMOpt lr=0.01'], fit_data_mnist, "Performance on mnist")
