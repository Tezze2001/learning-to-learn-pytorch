import csv
import os
import numpy as np
import torch
import torch.optim as optim
from torch.autograd.variable import Variable
from impl.optimizer import Optimizer
from impl.regularizers import EarlyStopping
import joblib
from tqdm import tqdm
import copy

cache = joblib.Memory(location='_cache', verbose=0)
USE_CUDA = torch.cuda.is_available()

def move_to_cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v

def detach_var(v):
    var = move_to_cuda(Variable(v.data, requires_grad=True))
    var.retain_grad()
    return var

def fit_outer_model(outer_model, outer_opt, inner_loss, inner_model_class, unroll = 20, n_steps_to_optimize = 100, out_mul = 1.0, should_train=0):
    """ 
        args:
            outer_model: model of neural optimizer
            outer_opt: meta-optimizer of neural optimizer
            inner_loss: loss function of inner problem
            inner_model_class: model of the inner problem
            unroll: number of optimization steps of inner loss to pass before update the neural optimizer weights
            n_steps_to_optimize: number of steps to optimize the inner loss
            out_mul: output multiplyer for adjusting output of neural optimizer
            should_train: 0 if you want to train the neural optimizer, 
        return:
            list of all losses

    """
    if should_train == 0:
        outer_model.train()
    else:
        outer_model.eval()
        unroll = 1
    
    target = inner_loss(training=should_train)
    optimizee = move_to_cuda(inner_model_class())
    n_params = 0
    for p in optimizee.parameters():
        n_params += int(np.prod(p.size()))
    hidden_states = [move_to_cuda(Variable(torch.zeros(n_params, outer_model.hidden_sz))) for _ in range(2)]
    cell_states = [move_to_cuda(Variable(torch.zeros(n_params, outer_model.hidden_sz))) for _ in range(2)]
    all_losses_ever = []
    if should_train == 0:
        outer_opt.zero_grad()
    all_losses = None
    for iteration in range(1, n_steps_to_optimize + 1):
        loss = optimizee(target)
                    
        if all_losses is None:
            all_losses = loss
        else:
            all_losses += loss
        
        all_losses_ever.append(loss.data.cpu().numpy())
        loss.backward(retain_graph=True if should_train == 0 else False)

        offset = 0
        result_params = {}
        hidden_states2 = [move_to_cuda(Variable(torch.zeros(n_params, outer_model.hidden_sz))) for _ in range(2)]
        cell_states2 = [move_to_cuda(Variable(torch.zeros(n_params, outer_model.hidden_sz))) for _ in range(2)]
        for name, p in optimizee.all_named_parameters():
            cur_sz = int(np.prod(p.size()))
            # We do this so the gradients are disconnected from the graph but we still get
            # gradients from the rest
            gradients = detach_var(p.grad.view(cur_sz, 1))
            updates, new_hidden, new_cell = outer_model(
                gradients,
                [h[offset:offset+cur_sz] for h in hidden_states],
                [c[offset:offset+cur_sz] for c in cell_states]
            )
            for i in range(len(new_hidden)):
                hidden_states2[i][offset:offset+cur_sz] = new_hidden[i]
                cell_states2[i][offset:offset+cur_sz] = new_cell[i]
            result_params[name] = p + updates.view(*p.size()) * out_mul
            result_params[name].retain_grad()
            
            offset += cur_sz
            
        if iteration % unroll == 0:
            if should_train == 0:
                outer_opt.zero_grad()
                all_losses.backward()
                outer_opt.step()
                
            all_losses = None
                        
            optimizee = move_to_cuda(inner_model_class(**{k: detach_var(v) for k, v in result_params.items()}))
            hidden_states = [detach_var(v) for v in hidden_states2]
            cell_states = [detach_var(v) for v in cell_states2]
            
        else:
            optimizee = move_to_cuda(inner_model_class(**result_params))
            assert len(list(optimizee.all_named_parameters()))
            hidden_states = hidden_states2
            cell_states = cell_states2
            
    return all_losses_ever

@cache.cache
def train_outer_model(inner_loss, inner_model_class, preproc=False, unroll=20, n_steps_to_optimize=100, n_epochs=20, n_tests=100, lr=0.001, out_mul=1.0, batch_size = 20):
    """
        inner_loss: loss function of inner problem
        inner_model_class: python class for the inner problem model
        unroll: number of optimization steps of inner loss to pass before update the meta optimizer weights
        n_steps_to_optimize: number of steps to optimize the inner loss
        n_epochs: number of epochs
        n_tests: number of function test on which test the meta optimizer
        lr: learning rate of the optimizer for meta optimizer
        out_mul: output multiplyer for adjusting output of meta optimizer
        returns:
            best validation loss, best optimizer patameters, 
    """
    opt_net = move_to_cuda(Optimizer(preproc=preproc))
    meta_opt = optim.Adam(opt_net.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    
    best_meta_net = None
    best_validation_loss = 100000000000000000

    training_loss_trajectory = []
    validation_loss_trajectory = []
    for _ in tqdm(range(n_epochs), 'Number of epochs'):
        training_loss = (np.mean([
            np.sum(fit_outer_model(opt_net, meta_opt, inner_loss, inner_model_class, unroll, n_steps_to_optimize, out_mul, should_train=0))
            for _ in tqdm(range(batch_size), f'Batch of function to train ({batch_size} funcs)')
        ]))

        training_loss_trajectory.append(training_loss)

        validation_loss = (np.mean([
            np.sum(fit_outer_model(opt_net, meta_opt, inner_loss, inner_model_class, unroll, n_steps_to_optimize, out_mul, should_train=1))
            for _ in tqdm(range(n_tests), f'Batch of function to validate ({n_tests} funcs)')
        ]))

        validation_loss_trajectory.append(validation_loss)

        if validation_loss < best_validation_loss:
            print(best_validation_loss, validation_loss)
            best_validation_loss = validation_loss
            best_meta_net = copy.deepcopy(opt_net.state_dict())

        early_stopping(validation_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
            
    return best_validation_loss, best_meta_net, training_loss_trajectory, validation_loss_trajectory

def grid_search_lr_meta_opt(log_file, inner_loss, inner_model, lr_values = [1.0, 0.1, 0.01, 0.001, 0.001, 0.0001], preproc=False):
    if not os.path.isfile(log_file):
        with open(log_file, mode="w", newline="") as file:
            writer = csv.writer(file, delimiter='|')
            writer.writerow(["learning_rate", "best_val", "train_loss_trajectory", "val_loss_trajectory"])
            # Grid search 
            for lr in tqdm(lr_values, 'Grid search on learning rate for the outer model optimizer'):
                print('Trying lr:', lr)
                if preproc:
                    out_mul=0.1
                else:
                    out_mul=1
                best_validation_loss, _, train_loss_trajectory, val_loss_trajectory = train_outer_model(inner_loss, inner_model, lr=lr, out_mul=out_mul, preproc=preproc)
                print('Best val loss:', best_validation_loss)
                writer.writerow([lr, best_validation_loss, train_loss_trajectory, val_loss_trajectory])
        print("End grid search")
    else:
        print('Log file already exists, grid search is not performed')

@cache.cache
def fit_std_optimizer(inner_loss, inner_model_class, std_opt_class, n_tests=100, n_steps_to_opt=100, **kwargs):
    results = []
    for i in tqdm(range(n_tests), 'Optimize a batch of functions'):
        target = inner_loss(training=False)
        optimizee = move_to_cuda(inner_model_class())
        optimizer = std_opt_class(optimizee.parameters(), **kwargs)
        total_loss = []
        for _ in range(n_steps_to_opt):
            loss = optimizee(target)
            
            total_loss.append(loss.data.cpu().numpy())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        results.append(total_loss)
    return results

def grid_search_lr_std_opt(log_file, inner_loss, inner_model_class, lr_params = [0.1, 0.01, 0.005, 0.001,  0.0005, 0.0001, 0.00005, 0.00001]):
    if not os.path.isfile(log_file):
        with open(log_file, mode="w", newline="") as file:
            writer = csv.writer(file, delimiter='|')
            writer.writerow(["optimizer", "learning_rate", "loss_trajectory"])
            # Grid search 
            NORMAL_OPTS = [(optim.Adam, {}), (optim.RMSprop, {}), (optim.SGD, {}), (optim.SGD, {'nesterov': True, 'momentum': 0.9})]
            OPT_NAMES = ['ADAM', 'RMSprop', 'SGD', 'NAG']

            # NB: the momentum parameter for nesterov was found from the following file: https://github.com/torch/optim/blob/master/nag.lua
            # since it is mentioned in the paper that "When an optimizer has more parameters than just a learning rate (e.g. decay coefficients for ADAM) we use the default values from the optim package in Torch7."
            for idx, (opt, kwargs) in enumerate(NORMAL_OPTS):

                best_loss = 1000000000000000.0
                best_lr = 0.0
                for lr in tqdm(lr_params, 'Learning rates'):
                    try:
                        loss = best_loss + 1.0
                        loss = np.mean([np.sum(s) for s in fit_std_optimizer(inner_loss, inner_model_class, opt, lr=lr, **kwargs)])
                        writer.writerow([OPT_NAMES[idx], lr, loss])
                    except RuntimeError:
                        pass
                    if loss < best_loss:
                        best_loss = loss
                        best_lr = lr

                print(f"Model: {OPT_NAMES[idx]}")
                print(f"best_lr: {best_lr}")
                print(f"best_loss: {best_loss}")

        print("End grid search")
    else:
        print('Log file already exists, grid search is not performed')