"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.

Part of this source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_laplace

def nrmse(xpred, xreal, eps=1e-8):
    assert xpred.ndim == xreal.ndim == 4, "number of dimensions for inputs must be 4!"

    diff = xpred - xreal
    num = torch.linalg.norm(diff.view(diff.size(0), -1), dim=1)
    den = torch.linalg.norm(xreal.view(xreal.size(0), -1), dim=1)
    return num / (den + eps)


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int | tuple[int, int] = 7, k1: float = 0.01, k2: float = 0.03, mode: str = '3d'):
        """
        Args:
            win_size: Window size for SSIM calculation. If `mode` is '3d', two
                values may be given as a tuple to denote differing window sizes
                in spatial and temporal dimensions.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
            mode: Convolution mode for SSIM calculation ('2d' or '3d'). If '2d'
                and input has a temporal dimension, the SSIM is calculated for
                each frame and averaged. If '3d', the SSIM is calculated for
                the whole volume.
        """
        super().__init__()
        if isinstance(win_size, int):
            self.win_size = [win_size]
        else:
            self.win_size = win_size
        if mode == '3d' and len(self.win_size) == 1:
            self.win_size = 2 * self.win_size
        self.k1, self.k2 = k1, k2

        if mode == '2d':
            w = torch.ones(1, 1, 1, self.win_size[0], self.win_size[0])
        elif mode == '3d':
            w = torch.ones(1, 1, self.win_size[1], self.win_size[0], self.win_size[0])
        else:
            raise ValueError(f'Unsupported mode {mode}')
        NP = w.numel()
        self.register_buffer('w', w / NP)
        self.cov_norm = NP / (NP - 1)

    def forward(self, pred: torch.Tensor, targ: torch.Tensor, data_range: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.w, torch.Tensor)
        assert pred.ndim == 5 and targ.ndim == 5, 'Input tensors must have 5 dimensions'

        if torch.is_complex(pred):
            pred = torch.abs(pred)
        if torch.is_complex(targ):
            targ = torch.abs(targ)

        data_range = data_range[:, None, None, None, None].type(pred.dtype)
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2

        ux = F.conv3d(pred, self.w)  # typing: ignore
        uy = F.conv3d(targ, self.w)  #
        uxx = F.conv3d(pred * pred, self.w)
        uyy = F.conv3d(targ * targ, self.w)
        uxy = F.conv3d(pred * targ, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        S_loss = 1 - S

        return S_loss.mean()