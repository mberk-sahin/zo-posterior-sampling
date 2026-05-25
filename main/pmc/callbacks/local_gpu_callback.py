import torch, pathlib, os, warnings, time #ehtplot.color
from typing import Optional, Iterable
from .base import BaseCallbackModule
from pmc.utils.normalize_image import normalize_image
from pmc.utils.save_image import save_image


class LocalGPUCallbackModule(BaseCallbackModule):
    
    def __init__(self) -> None:
        super().__init__()

    def on_batch_start(self, module, batch, batch_idx):
        # initialize memory lists
        self.start_mem = []
        self.start_mem_max = []
        self.end_mem = []
        self.end_mem_max = []

    def on_iteration_start(self, module, batch, batch_idx, t):
        # reset the max memory allocated
        torch.cuda.reset_max_memory_allocated()
        start_mem_val = torch.cuda.memory_allocated() / 1024**3 # GB
        start_mem_max_val = torch.cuda.max_memory_allocated() / 1024**3 # GB
        self.start_mem.append(start_mem_val)
        self.start_mem_max.append(start_mem_max_val)
        
    def on_iteration_end(self, module, iteration_outputs, batch, batch_idx, t):
        end_mem_val = torch.cuda.memory_allocated() / 1024**3 # GB
        end_mem_max_val = torch.cuda.max_memory_allocated() / 1024**3 # GB
        self.end_mem.append(end_mem_val)
        self.end_mem_max.append(end_mem_max_val)
    
    def on_batch_end(self, module, samples, means, stds, batch, batch_idx):
        module.logger.log_tensor({
            f'batch{batch_idx}_start_mem': torch.tensor(self.start_mem), 
            f'batch{batch_idx}_end_mem': torch.tensor(self.end_mem), 
            f'batch{batch_idx}_start_mem_max': torch.tensor(self.start_mem_max), 
            f'batch{batch_idx}_end_mem_max': torch.tensor(self.end_mem_max)
        })