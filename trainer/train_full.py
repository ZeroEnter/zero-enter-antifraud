import json
import os
import shutil

import torch
from ezkl import ezkl
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

    features = torch.tensor(features, dtype=torch.float)
    targets = torch.tensor(targets, dtype=torch.float)

    mean, std = features.mean([0,]), features.std(
        [
            0,
        ]
    )
    # torch.save(mean, "weights/mean.pt")
    # torch.save(std, "weights/std.pt")
    # features = (features - mean) / std
    print(f"mean: {mean}, std: {std}")

    assert (
        features.shape[0] == targets.shape[0]
    ), f"features.shape[0] != targets.shape[0], {features.shape[0]} != {targets.shape[0]}"

    model = SimpleAntiFraudGNN()
    model = simple_train_loop(
        num_epochs=201, model=model, features=features, targets=targets
    )
    model.eval()

    zkp_dir = "../ezkl_inference/data_zkp"
    os.makedirs(zkp_dir, exist_ok=True)
    shutil.rmtree(zkp_dir)
    os.makedirs(zkp_dir, exist_ok=True)

    model_path = os.path.join(zkp_dir, "network.onnx")
    compiled_model_path = os.path.join(zkp_dir, "network.compiled")
    pk_path = os.path.join(zkp_dir, "test.pk")
    vk_path = os.path.join(zkp_dir, "test.vk")
    settings_path = os.path.join(zkp_dir, "settings.json")
    srs_path = os.path.join(zkp_dir, "kzg.srs")
    witness_path = os.path.join(zkp_dir, "witness.json")
    data_path = os.path.join(zkp_dir, "input.json")
    proof_path = os.path.join(zkp_dir, "test.pf")

    torch.save(mean, "../weights/mean.pt")
    torch.save(std, "../weights/std.pt")
    features = (features - mean) / std

    # Export the model
    torch.onnx.export(
        model,  # model being run
        features,  # model input (or a tuple for multiple inputs)
        model_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )

    data_array = features.detach().numpy().tolist()

    data = dict(input_data=data_array)

    # Serialize data into file:
    json.dump(data, open(data_path, "w"))

    res = ezkl.gen_settings(model_path, settings_path)
    assert res == True

    res = ezkl.compile_model(model_path, compiled_model_path, settings_path)
    assert res == True

    # srs path
    res = ezkl.get_srs(srs_path, settings_path)
    assert res == True

    # now generate the witness file

    res = ezkl.gen_witness(
        data_path, compiled_model_path, witness_path, settings_path=settings_path
    )
    print(res)
    assert os.path.isfile(witness_path)

    # HERE WE SETUP THE CIRCUIT PARAMS
    # WE GOT KEYS
    # WE GOT CIRCUIT PARAMETERS
    # EVERYTHING ANYONE HAS EVER NEEDED FOR ZK

    res = ezkl.setup(
        compiled_model_path,
        vk_path,
        pk_path,
        srs_path,
        settings_path,
    )

    assert res == True
    assert os.path.isfile(vk_path)
    assert os.path.isfile(pk_path)
    assert os.path.isfile(settings_path)

    # GENERATE A PROOF
    res = ezkl.prove(
        witness_path,
        compiled_model_path,
        pk_path,
        proof_path,
        srs_path,
        "evm",
        "single",
        settings_path,
    )

    print(res)
    assert os.path.isfile(proof_path)

    # VERIFY IT

    res = ezkl.verify(
        proof_path,
        settings_path,
        vk_path,
        srs_path,
    )

    assert res == True
    print("verified")

    return model


if __name__ == "__main__":
    train()
