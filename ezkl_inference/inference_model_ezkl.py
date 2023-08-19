import json
import os
import shutil

import pandas as pd
import torch
from ezkl import ezkl

from models import SimpleAntiFraudGNN
from preprocessing import create_graph_dataset


def inference_ekzl(features, device):
    features = torch.tensor(features, dtype=torch.float)
    print(f"features.shape: {features.shape}")

    zkp_dir = "ezkl_inference/data_zkp"
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

    model = SimpleAntiFraudGNN(input_dim=features.shape[1], hidden_dim=16)
    dir2save_model = "weights"
    path2save_weights = os.path.join(
        dir2save_model, f"model_{model.__class__.__name__}.pth"
    )
    model.load_state_dict(
        torch.load(path2save_weights, map_location=device)
    )  # Choose whatever GPU device number you want
    model.to(device)

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

    data_array = ((features).detach().numpy()).reshape([-1]).tolist()

    data = dict(input_data=[data_array])

    # Serialize data into file:
    json.dump(data, open(data_path, "w"))

    res = ezkl.gen_settings(model_path, settings_path)
    assert res == True

    res = ezkl.compile_model(model_path, compiled_model_path, settings_path)
    assert res == True

    # srs path
    res = ezkl.get_srs(srs_path, settings_path)

    # now generate the witness file

    res = ezkl.gen_witness(
        data_path, compiled_model_path, witness_path, settings_path=settings_path
    )
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

    proof_path = os.path.join("test.pf")

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


def preproc_data_features():
    path2save_test_df = "data/preprocessed_test_set_credit_card_transactions-ibm_v2.csv"

    print(f"load test_df_set to: {os.path.basename(path2save_test_df)}")
    test_df_set = pd.read_csv(path2save_test_df)
    features, targets = create_graph_dataset(
        df=test_df_set,
    )
    return features
