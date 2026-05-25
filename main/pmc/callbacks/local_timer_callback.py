import torch, pathlib, os, warnings, time #ehtplot.color
from typing import Optional, Iterable
from .base import BaseCallbackModule
from pmc.utils.normalize_image import normalize_image
from pmc.utils.save_image import save_image


class LocalTimerCallbackModule(BaseCallbackModule):
    
    def __init__(self) -> None:
        super().__init__()
    
    def on_batch_start(self, module, batch, batch_idx):
        self.iter_durations = []

    def on_iteration_start(self, module, batch, batch_idx, t):
        self.start = time.time()       
        
    def on_iteration_end(self, module, iteration_outputs, batch, batch_idx, t):
        end = time.time()
        duration = end - self.start # seconds
        self.iter_durations.append(duration)
    
    def on_batch_end(self, module, samples, means, stds, batch, batch_idx):
        module.logger.log_tensor({
            f'batch{batch_idx}_iter_durations': torch.tensor(self.iter_durations), 
        })