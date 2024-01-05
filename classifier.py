import torch.nn as nn
import torch
from utils.model import RandPointCNN
from utils.util_funcs import knn_indices_func_gpu
from utils.util_layers import Dense

import utils.util_funcs
import utils.util_layers


# C_in, C_out, D, N_neighbors, dilution, N_rep, r_indices_func, C_lifted = None, mlp_width = 2
# (a, b, c, d, e) == (C_in, C_out, N_neighbors, dilution, N_rep)
# Abbreviated PointCNN constructor.
AbbPointCNN = lambda a, b, c, d, e: RandPointCNN(a, b, 3, c, d, e, knn_indices_func_gpu)



class Classifier(nn.Module):

    def __init__(self, output_channels=40):
        super(Classifier, self).__init__()

        self.pcnn1 = AbbPointCNN(3, 32, 8, 1, -1)
        self.pcnn2 = nn.Sequential(
            AbbPointCNN(32, 64, 8, 2, -1),
            AbbPointCNN(64, 96, 8, 4, -1),
            AbbPointCNN(96, 128, 12, 4, 120),
            AbbPointCNN(128, 160, 12, 6, 120)
        )

        self.fcn = nn.Sequential(
            Dense(160, 128),
            Dense(128, 64, drop_rate=0.5),
            Dense(64, output_channels, with_bn=False, activation=None)
        )

    def forward(self, x):
        device = x.device

        if type(x) != tuple:
            x = (x, torch.zeros(size=x.size()).to(device))

        x = self.pcnn1(x)
        x = self.pcnn2(x)[1]

        logits = self.fcn(x)
        logits_mean = torch.mean(logits, dim=1)
        return logits_mean
