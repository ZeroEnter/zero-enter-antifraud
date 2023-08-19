import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preproc_ibm_df(
    path_csv: str = "data/credit_card_transactions-ibm_v2.csv",
    n_samples: int = None,
    random_state: int = 42,
) -> pd.DataFrame:
    # Load the tabular dataset
    if n_samples is None:
        df = pd.read_csv(path_csv)
    else:
        df = pd.read_csv(path_csv).sample(n=n_samples, random_state=random_state)

    # The card_id is defined as one card by one user.
    # A specific user can have multiple cards, which would correspond to multiple different card_ids for this graph.
    # For this reason we will create a new column which is the concatenation of the column User and the Column Card
    df["card_id"] = df["User"].astype(str) + "_" + df["Card"].astype(str)

    # We need to strip the ‘$’ from the Amount to cast as a float
    df["Amount"] = df["Amount"].str.replace("$", "").astype(float)

    # time can't be casted to int so so opted to extract the hour and minute
    df["Hour"] = df["Time"].str[0:2]
    df["Minute"] = df["Time"].str[3:5]

    # drop unnecessary columns
    df = df.drop(["Time", "User", "Card"], axis=1)

    # ERRORS:
    # array([nan, 'Bad PIN', 'Insufficient Balance', 'Technical Glitch',
    #        'Bad Card Number', 'Bad CVV', 'Bad Expiration', 'Bad Zipcode',
    #        'Insufficient Balance,Technical Glitch', 'Bad Card Number,Bad CVV',
    #        'Bad CVV,Insufficient Balance',
    #        'Bad Card Number,Insufficient Balance'], dtype=object)

    df["Errors?"] = df["Errors?"].fillna("No error")

    # The two columns Zip and Merchant state contains missing values which can affect our graph.
    # Moreover these information can be extracted from the column Merchant City so we will drop them.
    df = df.drop(columns=["Merchant State", "Zip"], axis=1)

    # change the is fraud column to binary
    df["Is Fraud?"] = df["Is Fraud?"].apply(lambda x: 1 if x == "Yes" else 0)

    df["Merchant City"] = LabelEncoder().fit_transform(df["Merchant City"])

    # USE CHIP:
    # array(['Chip Transaction', 'Online Transaction', 'Swipe Transaction'],
    #       dtype=object)
    df["Use Chip"] = LabelEncoder().fit_transform(df["Use Chip"])
    df["Errors?"] = LabelEncoder().fit_transform(df["Errors?"])
    return df
