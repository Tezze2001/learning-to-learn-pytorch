import torch
import torch.nn as nn
from torch.autograd.variable import Variable 

from impl.utils import move_to_cuda


class QuadraticLoss:
    def __init__(self, **kwargs):
        self.W = move_to_cuda(Variable(torch.randn(10, 10)))
        self.y = move_to_cuda(Variable(torch.randn(10)))
        
    def get_loss(self, theta):
        return torch.sum((self.W.matmul(theta) - self.y)**2)
    
class QuadOptimizee(nn.Module):
    def __init__(self, theta=None):
        super().__init__()
        # Note: assuming the same optimization for theta as for
        # the function to find out itself.
        if theta is None:
            self.theta = nn.Parameter(torch.zeros(10))
        else:
            self.theta = theta
        
    def forward(self, target):
        return target.get_loss(self.theta)
    
    def all_named_parameters(self):
        return [('theta', self.theta)]