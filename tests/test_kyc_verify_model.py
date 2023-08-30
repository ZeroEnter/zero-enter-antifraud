import os

import pandas as pd
import torch
from torch import nn

from trainer import simple_train_loop
from models import SimpleAntiFraudGNN
from preprocessing import preproc_ibm_df, create_graph_dataset
import torch.nn.functional as F


def test():

    targets = torch.tensor(targets, dtype=torch.float32)


    model = SimpleAntiFraudGNN()
    model.to(device)

    model.eval()
    output = model(features)
    # criterion = nn.BCEWithLogitsLoss()
    # # Compute the loss
    # loss = criterion(output, targets)
    # print(loss)
    return output


if __name__ == "__main__":
    test()
