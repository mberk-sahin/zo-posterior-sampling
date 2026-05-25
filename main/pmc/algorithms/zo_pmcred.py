import torch
import torch.nn as nn
from .base import BaseAutoDM
import math
from typing import Optional
from .pmcred import PMCRED

class ZOPMCRED(PMCRED):

    def __init__(
        self,
        p: float,
        b_small: int, 
        b_large: Optional[int]=0,
        mu: Optional[float]=1e-5, 
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        
        # zeroth-order parameters
        self.mu = mu
        self.b_small = b_small
        self.b_large = b_large # 0 means use exact gradients
        self.gcurr = None
        self.xprev = None
        self.p = p

    def __call__(self, x, y, t, tmax):
        drift, df_grad, score = self.drift(x, y, t)
        xnextdrift = x + drift
        diffusion = self.diffusion(x, t)
        xnext = xnextdrift + diffusion

        # reset the gcurr and xprev for next sampling steps
        if t >= tmax-1:
            self.gcurr, self.xprev = None, None

        return xnext, xnextdrift, x, drift, score, diffusion, df_grad

    def drift(self, x, y, t):
        '''
        The iterate x has the following size 
        [B, C, H, W]
        '''
        # get gradient of the forward model
        df_grad = self.gcurr_update(x, y, t)
        # transform
        if self.transform is not None:
            x = self.transform(x)
        # compute the score
        sigma = self.coeff.score_coeff(self, x, t)
        # switch to evaluation mode
        if self.alpha == 0:
            score = torch.zeros_like(x)
        else:
            self.score_fn.eval()
            #self.score_fn = nn.DataParallel(self.score_fn).to(self.device)
            with torch.no_grad():
                alpha = max(self.alpha * sigma ** 2, 1)
                score = alpha * self.score_fn(
                                x, sigma * torch.ones(x.shape[0])
                            )           
        # combine to get the drift (Note the output of the score_fn is negative score)
        drift = self.gamma*(-df_grad + score)
        return drift, df_grad, score
    
    def gcurr_update(self, x, y, t):
        assert x.ndim == 4, 'x should have batch dimension'
        num_samples = len(x)
        die = torch.rand(num_samples)
        mask_batch_large = self.p > die if self.gcurr is not None else torch.ones(num_samples, dtype=torch.bool)
        
        batch_large_idxs = mask_batch_large.nonzero()
        batch_small_idxs = (~mask_batch_large).nonzero()

        x_batch_large, y_batch_large = x[batch_large_idxs].squeeze(dim=1), y[batch_large_idxs].squeeze(dim=1)
        x_batch_small, y_batch_small = x[batch_small_idxs].squeeze(dim=1), y[batch_small_idxs].squeeze(dim=1)
        
        gcurr = torch.zeros_like(x)
        if len(x_batch_large) != 0:
            if self.b_large == 0:
                update = self.forward_model.grad(x_batch_large, y_batch_large)
            else:
                update = self.forward_model.zograd(x_batch_large, y_batch_large, self.mu, batch_size=self.b_large)
            gcurr[batch_large_idxs.squeeze(dim=-1)] = update

        if len(x_batch_small) != 0:
            xprev_batch_small = self.xprev[batch_small_idxs.squeeze(dim=-1)]
            zo_grad_diff = self.forward_model.zo_grad_diff(x_batch_small, xprev_batch_small, y_batch_small, self.mu, batch_size=self.b_small)
            gcurr[batch_small_idxs.squeeze(dim=-1)] += zo_grad_diff

        # save x for future iterations
        self.xprev = x.clone()
        self.gcurr = gcurr.clone()
        return gcurr

