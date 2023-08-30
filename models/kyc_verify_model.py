from typing import Dict, List

import torch
from torch import nn


class SimpleKYC(nn.Module):
    # define nn
    def __init__(self):
        super(SimpleKYC, self).__init__()
        self.country_year = nn.Parameter(
            torch.tensor([[21.0, 0.0], [18.0, 1.0]], requires_grad=True)
        )

    def forward(self, x):
        """

        :param x: torch tensor array: []
        :return:
        """
        output = torch.zeros(x.shape[0])
        for cond_i in range(self.country_year.shape[0]):
            x_year = torch.where(
                x[:, 0] > self.country_year[cond_i, 0], 1.0, 0.0
            )
            x_country = torch.where(
                x[:, 1] == self.country_year[cond_i, 1], 1.0, 0.0
            )
            x_res = x_year * x_country
            output = x_res + output
            output = torch.where(
                output > 0.0, 1.0, 0.0
            )
        return output[None, :]
