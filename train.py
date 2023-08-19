import os

from sklearn.model_selection import train_test_split

from trainer import simple_train_loop
from models import SimpleAntiFraudGNN
from preprocessing import preproc_ibm_df, create_graph_dataset


def train():
    path2save_test_df = "data/preprocessed_test_set_credit_card_transactions-ibm_v2.csv"
    path2df_pandas_ibm = "data/credit_card_transactions-ibm_v2.csv"

    print(f"start preproc df: {os.path.basename(path2df_pandas_ibm)}")
    df_preprocessed = preproc_ibm_df(path_csv=path2df_pandas_ibm, n_samples=100000)
    print(df_preprocessed.head())
    train_df_set, test_df_set = train_test_split(
        df_preprocessed, test_size=0.1, random_state=42
    )

    print(f"save test_df_set to: {os.path.basename(path2save_test_df)}")
    test_df_set.to_csv(path2save_test_df, index_label=False)

    features, targets = create_graph_dataset(
        df=train_df_set,
    )

    assert (
        features.shape[0] == targets.shape[0]
    ), f"features.shape[0] != targets.shape[0], {features.shape[0]} != {targets.shape[0]}"

    model = SimpleAntiFraudGNN(input_dim=features.shape[1], hidden_dim=16)
    model = simple_train_loop(num_epochs=201, model=model, features=features, targets=targets)
    return model


if __name__ == "__main__":
    train()
