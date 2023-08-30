import os

import pandas as pd
import torch
from torch import nn

from trainer import simple_train_loop
from models import SimpleAntiFraudGNN, SimpleKYC
from preprocessing import preproc_ibm_df, create_graph_dataset
import torch.nn.functional as F


def test():
    features = [[29, 0], [17, 1], [27, 1], [21, 1]]
    features = torch.tensor(features).float()

    model = SimpleKYC()

    model.eval()
    output = model(features)
    return output


if __name__ == "__main__":
    test()
