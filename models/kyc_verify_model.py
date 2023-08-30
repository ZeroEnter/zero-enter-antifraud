from typing import Dict, List

import torch
from torch import nn


class SimpleKYC(nn.Module):
    # define nn
    def __init__(self):
        super(SimpleKYC, self).__init__()

        country_year = [[22, 1], [18, 0]]
        self.country_year = torch.tensor(country_year).float()

    def forward(self, x):
        """

        :param x: torch tensor array: []
        :return:
        """
        output = torch.zeros(x.shape[0]).int()
        for cond_i in range(self.country_year.shape[0]):
            x_year = torch.where(x[:, 0] > self.country_year[cond_i, 0].squeeze(), 1, 0).int()
            x_country = torch.where(x[:, 1] == self.country_year[cond_i, 1].squeeze(), 1, 0).int()
            x_res = x_year * x_country
            output = torch.bitwise_or(x_res, output)
        return output

