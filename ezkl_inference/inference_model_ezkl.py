import binascii
import json
import os
import shutil

import pandas as pd
import torch
from ezkl import ezkl
from xrpl.models import Memo

from models import SimpleAntiFraudGNN
from preprocessing import create_graph_dataset
from ezkl_inference.create_model_data import *
import torch.nn.functional as F

zkp_dir = "ezkl_inference/data_zkp"
os.makedirs(zkp_dir, exist_ok=True)


def inference_ekzl():
    model_path = os.path.join(zkp_dir, "network.onnx")
    compiled_model_path = os.path.join(zkp_dir, "network.compiled")
    pk_path = os.path.join(zkp_dir, "test.pk")
    vk_path = os.path.join(zkp_dir, "test.vk")
    settings_path = os.path.join(zkp_dir, "settings.json")
    srs_path = os.path.join(zkp_dir, "kzg.srs")
    witness_path = os.path.join(zkp_dir, "witness.json")
    data_path = os.path.join(zkp_dir, "input.json")
    proof_path = os.path.join(zkp_dir, "test.pf")

    run_args = ezkl.PyRunArgs()
    run_args.input_visibility = "encrypted"
    run_args.param_visibility = "public"
    run_args.output_visibility = "public"

    res = ezkl.gen_settings(model_path, settings_path, py_run_args=run_args)
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
    res_proof = ezkl.prove(
        witness_path,
        compiled_model_path,
        pk_path,
        proof_path,
        srs_path,
        "evm",
        "single",
        settings_path,
    )

    print(res_proof)
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
    return True


def preproc_data_features():
    mean, std = torch.load("weights/mean.pt"), torch.load("weights/std.pt")
    path2save_test_df = "data/preprocessed_test_set_credit_card_transactions-ibm_v2.csv"

    print(f"load test_df_set to: {os.path.basename(path2save_test_df)}")
    test_df_set = pd.read_csv(path2save_test_df)
    features, targets = create_graph_dataset(
        df=test_df_set,
    )
    features = torch.tensor(features, dtype=torch.float32)
    features = (features - mean) / std
    return features


if __name__ == "__main__":
    # create_model_data()
    inference_ekzl()
