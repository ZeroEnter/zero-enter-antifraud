import json
from datetime import datetime

import requests
import base64
import os

import torch
from ezkl import ezkl
from tqdm import tqdm
import pandas as pd

from models import SimpleKYC

ZKP_DIR_STAT = "stat_res_dir"
os.makedirs(ZKP_DIR_STAT, exist_ok=True)

OUTPUT = dict()

data_KYC = [
    "[[29.0, 0.0]]",
    "[[20.0, 0.0]]",
    "[[20.0, 1.0]]",
    "[[27.0, 1.0]]",
    "[[17.0, 1.0]]",
]


def to_b64(path_to_file):
    with open(path_to_file, "rb") as f:
        file_content = f.read()

    base64_encoded = base64.b64encode(file_content).decode("utf-8")
    return base64_encoded


def test():
    headers = {"Content-Type": "application/json"}
    type_model = "simple_kyc"
    url = "http://localhost:8000/inference"
    model_path_pytorch = "weights/model_SimpleKYC.pth"

    for data_KYC_sample in tqdm(data_KYC):
        input_js = {
            "link_onnx": "https://raw.githubusercontent.com/ZeroEnter/zero-enter-antifraud/main/weights/model_SimpleKYC.pth",
            "type_model": type_model,
            "input_data": data_KYC_sample,
        }

        model = SimpleKYC()
        model.load_state_dict(
            torch.load(model_path_pytorch, map_location=torch.device("cpu"))
        )  # Choose whatever GPU device number you want
        model.to(torch.device("cpu"))

        features_ = json.loads(data_KYC_sample)
        te_x = torch.Tensor(features_).float()
        output = model(te_x)
        output = int(output.item())

        response_inference = requests.post(url, headers=headers, json=input_js)
        data = json.loads(response_inference.content)
        # Verify
        for field, input_data in data.items():
            base64_bytes = input_data.encode("utf-8")
            decoded_bytes = base64.b64decode(base64_bytes)
            # decoded_string = decoded_bytes.decode("utf-8")
            with open(os.path.join(ZKP_DIR_STAT, field), "wb") as f:
                f.write(decoded_bytes)

        st_datatime = datetime.now()
        # return {"result": bool}
        result = ezkl.verify(
            proof_path=os.path.join(ZKP_DIR_STAT, "test.pf"),
            settings_path=os.path.join(ZKP_DIR_STAT, "settings.json"),
            vk_path=os.path.join(ZKP_DIR_STAT, "test.vk"),
            srs_path=os.path.join(ZKP_DIR_STAT, "kzg.srs"),
        )
        time_spent = (st_datatime.now() - st_datatime).total_seconds()
        OUTPUT[data_KYC_sample] = [time_spent, result, output]

    df = pd.DataFrame.from_dict(
        OUTPUT, orient="index", columns=["time, seconds", "result_verify", "output"]
    )
    df["input_data"] = df.index

    df = df.reset_index()
    df = df.drop(columns=["index"])
    df.to_csv(
        os.path.join(ZKP_DIR_STAT, f"statistic_res_{type_model}.csv"), index=False
    )

    # with open(os.path.join(ZKP_DIR_STAT, f"statistic_res_{type_model}.json"), "w", encoding="utf8") as f:
    #     json.dump(OUTPUT, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    test()
