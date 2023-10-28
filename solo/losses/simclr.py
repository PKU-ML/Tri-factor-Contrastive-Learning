# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import torch.nn.functional as F
from solo.utils.misc import gather, get_rank


def simclr_loss_func(
    z: torch.Tensor, indexes: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.

    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: SimCLR loss.
    """

    gathered_z = gather(z)

    sim = torch.exp(torch.einsum("if, jf -> ij", z, gathered_z) / temperature)

    gathered_indexes = gather(indexes)

    indexes = indexes.unsqueeze(0)
    gathered_indexes = gathered_indexes.unsqueeze(0)
    # positives
    pos_mask = indexes.t() == gathered_indexes
    pos_mask[:, z.size(0) * get_rank() :].fill_diagonal_(0)
    # negatives
    neg_mask = indexes.t() != gathered_indexes

    pos = torch.sum(sim * pos_mask, 1)
    neg = torch.sum(sim * neg_mask, 1)
    loss = -(torch.mean(torch.log(pos / (pos + neg))))
    return loss

def simclr_loss_func_tri(
    z: torch.Tensor, scale_param, indexes: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.

    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: SimCLR loss.
    """
    N, D = z.size()

    z = F.normalize(z, dim=-1)
    gathered_z = gather(z)

    corr = z.T @ z / N
    diag = torch.eye(D, device=corr.device)
    cdif = (corr - diag).pow(2)
    dec_loss = cdif.mean()



    z1, z2 = torch.chunk(z, 2)
    scale = torch.nn.functional.softplus(scale_param).unsqueeze(0)
    z1 = z1 * scale
    z = torch.cat([z1, z2])

    sim = torch.exp(torch.einsum("if, jf -> ij", z, gathered_z) / temperature)

    gathered_indexes = gather(indexes)

    indexes = indexes.unsqueeze(0)
    gathered_indexes = gathered_indexes.unsqueeze(0)
    # positives
    pos_mask = indexes.t() == gathered_indexes
    pos_mask[:, z.size(0) * get_rank() :].fill_diagonal_(0)
    # negatives
    neg_mask = indexes.t() != gathered_indexes

    pos = torch.sum(sim * pos_mask, 1)
    neg = torch.sum(sim * neg_mask, 1)
    loss = -(torch.mean(torch.log(pos / (pos + neg))))


    return loss, dec_loss



def spectral(
    z: torch.Tensor, indexes: torch.Tensor, scale_param=None, temperature: float = 0.1
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.

    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: SimCLR loss.
    """
    N, D = z.size()
    # corr = (z.T @ z / N).clamp(-1e3, 1e3)

    z = F.normalize(z, dim=-1)
    gathered_z = gather(z)


    sim = torch.einsum("if, jf -> ij", z, gathered_z) / temperature
    # sim = torch.einsum("if, jf -> ij", z, gathered_z)

    gathered_indexes = gather(indexes)

    indexes = indexes.unsqueeze(0)
    gathered_indexes = gathered_indexes.unsqueeze(0)
    # positives
    pos_mask = indexes.t() == gathered_indexes
    pos_mask[:, z.size(0) * get_rank() :].fill_diagonal_(0)
    # negatives
    neg_mask = indexes.t() != gathered_indexes

    pos = torch.sum(sim * pos_mask, 1)
    neg = torch.mean((sim * neg_mask).pow(2), 1)
    spectral_loss = -torch.mean(pos) + torch.mean(neg)

    return spectral_loss


def spectral_tri(
    z: torch.Tensor, indexes: torch.Tensor, scale_param=None, temperature: float = 0.1
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z
    from different views, a positive boolean mask of all positives and
    a negative boolean mask of all negatives.

    Args:
        z (torch.Tensor): (N*views) x D Tensor containing projected features from the views.
        indexes (torch.Tensor): unique identifiers for each crop (unsupervised)
            or targets of each crop (supervised).

    Return:
        torch.Tensor: SimCLR loss.
    """
    N, D = z.size()

    z = F.normalize(z, dim=-1)
    gathered_z = gather(z)


    corr = z.T @ z
    diag = torch.eye(D, device=corr.device)
    cdif = (corr - diag).pow(2)
    dec_loss = cdif.mean()

    if True:
        z1, z2 = torch.chunk(z, 2)
        scale = torch.nn.functional.softplus(scale_param).unsqueeze(0)
        z1 = z1 * scale 
        z = torch.cat([z1, z2])

    sim = torch.einsum("if, jf -> ij", z, gathered_z)

    gathered_indexes = gather(indexes)

    indexes = indexes.unsqueeze(0)
    gathered_indexes = gathered_indexes.unsqueeze(0)
    # positives
    pos_mask = indexes.t() == gathered_indexes
    pos_mask[:, z.size(0) * get_rank() :].fill_diagonal_(0)
    # negatives
    neg_mask = indexes.t() != gathered_indexes

    pos = torch.sum(sim * pos_mask, 1)
    neg = torch.mean((sim * neg_mask).pow(2), 1)
    spectral_loss = -torch.mean(pos) + torch.mean(neg)


    return spectral_loss, dec_loss


