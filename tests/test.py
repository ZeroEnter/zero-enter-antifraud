import os

import pandas as pd
import torch
from torch import nn

from trainer import simple_train_loop
from models import SimpleAntiFraudGNN
from preprocessing import preproc_ibm_df, create_graph_dataset
import torch.nn.functional as F


def test():
    mean, std = torch.load('weights/mean.pt'), torch.load('weights/std.pt')
    path2save_test_df = "data/preprocessed_test_set_credit_card_transactions-ibm_v2.csv"
    d = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(d) if torch.cuda.is_available() else torch.device(d)

    print(f"load test_df_set to: {os.path.basename(path2save_test_df)}")
    test_df_set = pd.read_csv(path2save_test_df)

    features, targets = create_graph_dataset(
        df=test_df_set,
    )
    features = torch.tensor(features, dtype=torch.float32)
    features = (features - mean) / std
    targets = torch.tensor(targets, dtype=torch.float32)

    print(f"features.shape: {features.shape}")
    print(f"target: {targets.shape}")

    assert (
        features.shape[0] == targets.shape[0]
    ), f"features.shape[0] != targets.shape[0], {features.shape[0]} != {targets.shape[0]}"

    model = SimpleAntiFraudGNN()
    dir2save_model = "weights"
    path2save_weights = os.path.join(
        dir2save_model, f"model_{model.__class__.__name__}.pth"
    )
    model.load_state_dict(
        torch.load(path2save_weights, map_location=d)
    )  # Choose whatever GPU device number you want
    model.to(device)

    model.eval()
    output = model(features)
    criterion = nn.BCEWithLogitsLoss()
    # Compute the loss
    loss = criterion(output, targets)
    print(loss)
    return loss


if __name__ == "__main__":
    test()
