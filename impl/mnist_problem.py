import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd.variable import Variable
import torch.nn as nn
from torchvision import datasets
import torchvision

from impl.utils import move_to_cuda


class MNISTLoss:
    def __init__(self, setting=0):
        if setting in [0, 1]:
            dataset = datasets.MNIST(
                './mnist', train=True, download=True,
                transform=torchvision.transforms.ToTensor()
            )
        else: 
            dataset = datasets.MNIST(
                './mnist', train=False, download=True,
                transform=torchvision.transforms.ToTensor()
            )
            
        indices = list(range(len(dataset)))
        np.random.RandomState(10).shuffle(indices)
        if setting == 0:
            indices = indices[:len(indices) // 2]
        if setting == 1:
            indices = indices[len(indices) // 2:]

        self.loader = torch.utils.data.DataLoader(
            dataset, batch_size=128,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))
        
        self.batches = []
        self.cur_batch = 0
        
    def sample(self):
        if self.cur_batch >= len(self.batches):
            self.batches = []
            self.cur_batch = 0
            for b in self.loader:
                self.batches.append(b)
        batch = self.batches[self.cur_batch]
        self.cur_batch += 1
        return batch

class MNISTNet(nn.Module):
    def __init__(self, layer_size=20, n_layers=1, activation = nn.Sigmoid(), **kwargs):
        super().__init__()
        # Sadly this network needs to be implemented without using the convenient pytorch
        # abstractions such as nn.Linear, because afaik there is no way to load parameters
        # in those in a way that preserves gradients.
        if kwargs != {}:
            self.params = kwargs
        else:
            inp_size = 28*28
            self.params = {}
            for i in range(n_layers):
                self.params[f'mat_{i}'] = nn.Parameter(torch.randn(inp_size, layer_size))
                self.params[f'bias_{i}'] = nn.Parameter(torch.zeros(layer_size))
                inp_size = layer_size

            self.params['final_mat'] = nn.Parameter(torch.randn(inp_size, 10))
            self.params['final_bias'] = nn.Parameter(torch.zeros(10))
            
            # Put it all in a module list so that ordinary optimizers can find them.
            self.mods = nn.ParameterList()
            for v in self.params.values():
                self.mods.append(v)
                
        self.activation = activation
        self.loss = nn.NLLLoss()
                
    def all_named_parameters(self):
        return [(k, v) for k, v in self.params.items()]
    
    def forward(self, loss):
        inp, out = loss.sample()
        inp = move_to_cuda(Variable(inp.view(inp.size()[0], 28*28)))
        out = move_to_cuda(Variable(out))
        
        cur_layer = 0
        while f'mat_{cur_layer}' in self.params:
            inp = self.activation(torch.matmul(inp, self.params[f'mat_{cur_layer}']) + self.params[f'bias_{cur_layer}'])
            cur_layer += 1
                    
        inp = F.log_softmax(torch.matmul(inp, self.params['final_mat']) + self.params['final_bias'], dim=1)
        l = self.loss(inp, out)
        return l