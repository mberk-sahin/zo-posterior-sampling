import torch
import math
from abc import ABC, abstractmethod
from pmc.utils.add_awgn import addawgn
from typing import Callable, Optional
from torch import nn
import time

class IdentityAE(nn.Module):
    def encoder(self, x):
        return x
    def decoder(self, x):
        return x

class BaseForwardModel(ABC):
    def __init__(
        self,
        input_snr: float,
        var: float,
        autoencoder: Optional[Callable] = None,
        no_var_in_grad: Optional[bool] = False,
    ) -> None:
        self.input_snr = input_snr
        self.var = var
        self.autoencoder = autoencoder
        # this is for DPS because it uses step size instead of 1/var
        self.no_var_in_grad = no_var_in_grad

    def __call__(self, x):
        '''
        generate the noisy measurements y
        Args:
            x: input tensor with shape [B, C, H, W]
        '''
        y, _, noise_level = addawgn(self.forward(x), self.input_snr)
        print(f'Current noise level (beta=sqrt(variance)) is {noise_level}')
        return y

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def adjoint(self, y):
        pass   

    def grad(self, x, y):
        if self.no_var_in_grad:
            return self.adjoint(self.forward(x) - y).real # add real later
        else:
            return self.adjoint(self.forward(x) - y).real / self.var # add real  later
    
    def eval(self, x, y):
        return torch.linalg.matrix_norm(self.forward(x) - y, dim=(-1, -2), keepdim=True) ** 2 / (2 * self.var)
        #return (self.forward(x) - y).norm() ** 2 / (2 * self.var)
    
    @torch.no_grad()
    def generate_noise(self, x):
        if self.autoencoder is None:
            return torch.randn_like(x)
        else:
            device = x.device
            # sample noise in low dimensional space 32x32x4
            z = torch.randn(len(x), 4, 32, 32, device=device, dtype=x.dtype)
            decoded = self.autoencoder.decode(z).detach().clone().contiguous()
            return decoded

    def zograd(self, x, y, mu, batch_size=1, max_batch_size=50):

        if batch_size > max_batch_size:
            partitions = [max_batch_size] * (batch_size // max_batch_size) + ([batch_size % max_batch_size] if batch_size % max_batch_size != 0 else [])
        else: 
            partitions = [batch_size,]
        
        zo_grad = 0 
        x_shape, y_shape = x.shape, y.shape
        for batch_len in partitions:
            # Repeat input
            x_c = x.expand(batch_len, *x_shape).reshape(-1, *x_shape[1:]).contiguous()
            y_c = y.expand(batch_len, *y_shape).reshape(-1, *y_shape[1:]).contiguous()

            u = self.generate_noise(x_c)
            xperturbed = x_c + mu * u 
            zo_grads = (self.eval(xperturbed, y_c) - self.eval(x_c, y_c)) / mu * u
            zo_grad += zo_grads.reshape(batch_len, *x_shape).sum(dim=0) / batch_size
        return zo_grad
        
    def zo_grad_diff(self, x1, x2, y, mu, batch_size=1, max_batch_size=50):

        if batch_size > max_batch_size:
            partitions = [max_batch_size] * (batch_size // max_batch_size) + ([batch_size % max_batch_size] if batch_size % max_batch_size != 0 else [])
        else: 
            partitions = [batch_size,]

        zo_grad_diff = 0
        x_shape, y_shape = x1.shape, y.shape
        for batch_len in partitions:
            # Repeat inputs
            x1_c = x1.expand(batch_len, *x_shape).reshape(-1, *x_shape[1:]).contiguous()
            x2_c = x2.expand(batch_len, *x_shape).reshape(-1, *x_shape[1:]).contiguous()
            y_c = y.expand(batch_len, *y_shape).reshape(-1, *y_shape[1:]).contiguous()
            u = self.generate_noise(x1_c)
            
            x1_perturbed = x1_c + mu * u 
            zo_grads1 = (self.eval(x1_perturbed, y_c) - self.eval(x1_c, y_c)) / mu * u
            zo_grads1 = zo_grads1.reshape(batch_len, *x_shape).sum(dim=0) / batch_size
            # calculate the ZO for previous term
            x2_perturbed = x2_c + mu * u
            zo_grads2 = (self.eval(x2_perturbed, y_c) - self.eval(x2_c, y_c)) / mu * u
            zo_grads2 = zo_grads2.reshape(batch_len, *x_shape).sum(dim=0) / batch_size
            
            # calculate zeroth-order diff. update
            zo_grad_diff += (zo_grads1 - zo_grads2)
        
        return zo_grad_diff
