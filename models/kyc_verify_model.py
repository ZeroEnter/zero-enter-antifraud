from typing import Dict, List

import torch
from torch import nn


class SimpleKYC(nn.Module):
    # define nn
    def __init__(self):
        super(SimpleKYC, self).__init__()

        self.country_year = torch.tensor([[22.0, 1.0], [18.0, 0.0]])

    def forward(self, x):
        """

        :param x: torch tensor array: []
        :return:
        """
        x_year = torch.where(x > self.country_year[:, 0].squeeze(), 1.0, 0.0)
        x_country = torch.where(x == self.country_year[:, 1].squeeze(), 1.0, 0.0)
        x = x_year @ x_country
        return x
