import os

import torch
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

from models import SimpleAntiFraudGNN
from preprocessing import preproc_ibm_df, create_graph_dataset
from trainer import simple_train_loop


def train():
    path2save_test_df = "data/preprocessed_test_set_credit_card_transactions-ibm_v2.csv"
    path2df_pandas_ibm = "data/credit_card_transactions-ibm_v2.csv"

    print(f"start preproc df: {os.path.basename(path2df_pandas_ibm)}")
    df_preprocessed = preproc_ibm_df(path_csv=path2df_pandas_ibm, n_samples=100000)
    print(df_preprocessed.head())
    train_df_set, test_df_set = train_test_split(
        df_preprocessed, test_size=0.0001, random_state=42
    )


    print(f"save test_df_set to: {os.path.basename(path2save_test_df)}")
    test_df_set.to_csv(path2save_test_df, index=False)

    train_df_set, val_df_set = train_test_split(
        train_df_set, test_size=0.1, random_state=42
    )
    print(f"train_df_set: {len(train_df_set)}, val_df_set: {len(val_df_set)}, test_df_set: {len(test_df_set)}")

    train_X, train_y = create_graph_dataset(
        df=train_df_set,
    )
    test_X, test_y = create_graph_dataset(
        df=val_df_set,
    )

    # Convert training data to pytorch variables
    tr_x = torch.Tensor(train_X).float()
    # tr_x = (tr_x - tr_x.mean(dim=0)) / tr_x.std(dim=0)
    train_X = Variable(tr_x)

    te_x = torch.Tensor(test_X).float()
    # te_x = (te_x - te_x.mean(dim=0)) / te_x.std(dim=0)
    test_X = Variable(te_x)
    train_y = Variable(torch.Tensor(train_y).long())
    test_y = Variable(torch.Tensor(test_y).long())

    mean, std = train_X.mean([0,]), train_X.std(
        [
            0,
        ]
    )
    torch.save(mean, "weights/mean.pt")
    torch.save(std, "weights/std.pt")
    # features = (features - mean) / std
    # print(f"mean: {mean}, std: {std}")

    model = SimpleAntiFraudGNN()
    model = simple_train_loop(
        num_epochs=201, model=model, train_X=train_X, test_X=test_X, train_y=train_y, test_y=test_y
    )
    return model


if __name__ == "__main__":
    train()
