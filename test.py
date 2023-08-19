import os

import pandas as pd
import torch

from trainer import simple_train_loop
from models import SimpleAntiFraudGNN
from preprocessing import preproc_ibm_df, create_graph_dataset


def test():
    path2save_test_df = "data/preprocessed_test_set_credit_card_transactions-ibm_v2.csv"
    d = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(d) if torch.cuda.is_available() else torch.device(d)

    print(f"load test_df_set to: {os.path.basename(path2save_test_df)}")
    test_df_set = pd.read_csv(path2save_test_df)

    features, targets = create_graph_dataset(
        df=test_df_set,
    )
    features = torch.tensor(features, dtype=torch.float)
    targets = torch.tensor(targets, dtype=torch.float)

    print(f"features.shape: {features.shape}")
    print(f"target: {targets.shape}")

    assert (
        features.shape[0] == targets.shape[0]
    ), f"features.shape[0] != targets.shape[0], {features.shape[0]} != {targets.shape[0]}"

    model = SimpleAntiFraudGNN(input_dim=features.shape[1], hidden_dim=16)
    dir2save_model = "weights"
    path2save_weights = os.path.join(
        dir2save_model, f"model_{model.__class__.__name__}.pth"
    )
    model.load_state_dict(
        torch.load(path2save_weights, map_location=d)
    )  # Choose whatever GPU device number you want
    model.to(device)

    return model


if __name__ == "__main__":
    test()
