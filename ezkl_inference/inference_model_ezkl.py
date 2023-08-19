import os

import pandas as pd
import torch
from torch import nn

from trainer import simple_train_loop
from models import SimpleAntiFraudGNN
from preprocessing import preproc_ibm_df, create_graph_dataset


def inference_ekzl(features, device):
    features = torch.tensor(features, dtype=torch.float)
    print(f"features.shape: {features.shape}")

    model = SimpleAntiFraudGNN(input_dim=features.shape[1], hidden_dim=16)
    dir2save_model = "weights"
    path2save_weights = os.path.join(
        dir2save_model, f"model_{model.__class__.__name__}.pth"
    )
    model.load_state_dict(
        torch.load(path2save_weights, map_location=device)
    )  # Choose whatever GPU device number you want
    model.to(device)

    model.eval()
    output = model(features)
    print(output)
    return output


def preproc_data_features():
    path2save_test_df = "data/preprocessed_test_set_credit_card_transactions-ibm_v2.csv"

    print(f"load test_df_set to: {os.path.basename(path2save_test_df)}")
    test_df_set = pd.read_csv(path2save_test_df)
    features, targets = create_graph_dataset(
        df=test_df_set,
    )
    return features
