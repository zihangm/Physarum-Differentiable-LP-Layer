### required packages:
### Pytorch-1.0+ (if below 1.1, replace torch.solve with torch.gesv)
### scipy
### numpy

import torch
import torch.nn.functional as F
import numpy as np

def physarum_solve(A, b, c, step_size=1, max_iter = 10):
    ### INPUT ###
    ### A.shape: batch_size x m x n
    ### b.shape: batch_size x m x 1
    ### c.shape: batch_size x n
    ### OUTPUT ###
    ### xs.shape: batch_size x n

    batch_size = A.shape[0]
    n = A.shape[2]
    xs = torch.rand([batch_size, n]).cuda()
    for i in range(max_iter):
        W_diag = xs / c
        W = torch.diag_embed(W_diag)
        L = torch.matmul(torch.matmul(A, W), torch.transpose(A, 1, 2))
        p, _ = torch.solve(b, L)  # use torch.gesv to replace torch.solve if below Pytorch1.1
        q = torch.matmul(torch.matmul(W, torch.transpose(A, 1, 2)), p)
        xs = (1 - step_size) * xs + step_size * q.squeeze(2)
        xs = torch.clamp(xs, min=1e-6, max=1e+4)
    return xs
