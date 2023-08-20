import os

import torch
from sklearn.model_selection import train_test_split

from trainer import simple_train_loop
from models import SimpleAntiFraudGNN
from preprocessing import preproc_ibm_df, create_graph_dataset

import torch.nn.functional as F
from torchvision import transforms


def train():
    path2save_test_df = "data/preprocessed_test_set_credit_card_transactions-ibm_v2.csv"
    path2df_pandas_ibm = "data/credit_card_transactions-ibm_v2.csv"

    print(f"start preproc df: {os.path.basename(path2df_pandas_ibm)}")
    df_preprocessed = preproc_ibm_df(path_csv=path2df_pandas_ibm, n_samples=100000)
    print(df_preprocessed.head())
    train_df_set, test_df_set = train_test_split(
        df_preprocessed, test_size=0.1, random_state=42
    )
    print(f"train_df_set: {len(train_df_set)}, test_df_set: {len(test_df_set)}")

    print(f"save test_df_set to: {os.path.basename(path2save_test_df)}")
    test_df_set.to_csv(path2save_test_df, index_label=False)

    features, targets = create_graph_dataset(
        df=train_df_set,
    )

    features = torch.tensor(features, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    # targets = torch.tensor(targets, dtype=torch.long)

    mean, std = features.mean([0,]), features.std([0,])
    torch.save(mean, 'weights/mean.pt')
    torch.save(std, 'weights/std.pt')
    # features = (features - mean) / std
    # print(f"mean: {mean}, std: {std}")

    assert (
        features.shape[0] == targets.shape[0]
    ), f"features.shape[0] != targets.shape[0], {features.shape[0]} != {targets.shape[0]}"

    model = SimpleAntiFraudGNN()
    model = simple_train_loop(
        num_epochs=201, model=model, features=features, targets=targets
    )
    return model


if __name__ == "__main__":
    train()
