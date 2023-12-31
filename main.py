import base64
import binascii
import glob
import io
import json
import os
import shutil
from typing import List

import httpx
import pandas as pd
import requests
import torch
from ezkl import ezkl
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ezkl_inference import inference_ekzl, convert_model_data
from torch.autograd import Variable
import base64

from models import SimpleKYC


def string_to_hex(s):
    if isinstance(s, str):
        return binascii.hexlify(s.encode()).decode()
    return binascii.hexlify(s).decode()


def read_file_as_base64(filename):
    with open(filename, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


MAP_RES = {
    "US": 0.0,
    "Luxembourg": 1.0,
}

HOST = os.getenv("HOST", "localhost")
PORT = os.getenv("PORT", 8000)

zkp_dir = "ezkl_inference/data_zkp"
os.makedirs(zkp_dir, exist_ok=True)

app = FastAPI()

rename = {"test.vk": "vk", "test.pf": "proof"}


class Item(BaseModel):
    input_data: str
    type_model: str
    link_onnx: str


class ItemMain(BaseModel):
    input_data: List[Item]


async def verify(
    proof_path,
    settings_path,
    vk_path,
    srs_path,
):
    res = ezkl.verify(
        proof_path,
        settings_path,
        vk_path,
        srs_path,
    )
    return res


@app.post("/inference")
async def create_inference(item: ItemMain):
    input_data_main = item.input_data
    output_data_main = {"output": []}

    for input_data in input_data_main:
        type_model = input_data.type_model
        link_onnx = input_data.link_onnx
        model_path = os.path.join(zkp_dir, f"network_{type_model}.onnx")
        model_path_pytorch = os.path.join(zkp_dir, f"network_{type_model}.pth")
        data_path = os.path.join(zkp_dir, f"input_{type_model}.json")
        data_path_pre = os.path.join(zkp_dir, f"pre_input_{type_model}.json")

        vk_path = os.path.join(zkp_dir, f"test_{type_model}.vk")
        settings_path = os.path.join(zkp_dir, f"settings_{type_model}.json")
        srs_path = os.path.join(zkp_dir, f"kzg_{type_model}.srs")
        proof_path = os.path.join(zkp_dir, f"test_{type_model}.pf")

        async with httpx.AsyncClient() as client:
            response = await client.get(link_onnx)
            if response.status_code == 200:
                with open(model_path_pytorch, "wb") as f:
                    f.write(response.content)
            else:
                return {"error": f"Failed to download {link_onnx}"}

        if type_model == "anti_fraud":
            # base64_bytes = input_data.encode("utf-8")
            # decoded_bytes = base64.b64decode(base64_bytes)
            # decoded_string = decoded_bytes.decode("utf-8")
            # data = pd.read_csv(io.StringIO(decoded_string))
            data = pd.read_csv(input_data.input_data)

            data = data.iloc[:1, :]

            data_path = convert_model_data(
                test_df_set=data,
                model_path=model_path,
                data_path=data_path,
                model_path_pytorch=model_path_pytorch,
            )
            assert os.path.exists(data_path), f"not found {data_path}"

        elif type_model == "simple_kyc":
            async with httpx.AsyncClient() as client:
                response = await client.get(input_data.input_data)
                if response.status_code == 200:
                    with open(data_path_pre, "wb") as f:
                        f.write(response.content)
                else:
                    return {"error": f"Failed to download {link_onnx}"}

            with open(data_path_pre, "r") as f:
                features_ = json.load(f)

            features_pre = features_["input_data"]
            features_ = [
                [float(features_pre["age"]), MAP_RES[features_pre["residence"]]]
            ]

            device = torch.device("cpu")

            model = SimpleKYC()
            model.load_state_dict(
                torch.load(model_path_pytorch, map_location=device)
            )  # Choose whatever GPU device number you want
            model.to(device)

            te_x = torch.Tensor(features_).float()
            features = Variable(te_x)

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

            data = dict(input_data=features_)

            # Serialize data into file:
            json.dump(data, open(data_path, "w"))

        await inference_ekzl(
            vk_path=vk_path,
            settings_path=settings_path,
            srs_path=srs_path,
            proof_path=proof_path,
            data_path=data_path,
            model_path=model_path,
            type_model=type_model,
        )
        output_data_main["output"].append({
            "test.vk": read_file_as_base64(vk_path),
            "test.pf": read_file_as_base64(proof_path),
            "kzg.srs": read_file_as_base64(srs_path),
            "settings.json": read_file_as_base64(settings_path),
        })
    return output_data_main

    # with open(os.path.join(zkp_dir, f"test_{type_model}.vk"), "rb") as f:
    #     vk = f.read()

    # return {"files": {"proof": pf, "vk": read_file_as_base64(os.path.join(zkp_dir, f"test_{type_model}.vk"))}}
    # files_to_send = glob.glob(os.path.join(zkp_dir, "*"))
    # return {
    #     "files": {
    #         rename[f.split("/")[-1]]: read_file_as_base64(f)
    #         for f in files_to_send
    #         # if f.split("/")[-1] in ["test.vk", "test.pf", "kzg.srs", "settings.json"]
    #         if f.split("/")[-1] in ["test.vk", "test.pf"]
    #     }
    # }


#
# @app.post("/inference")
# async def create_inference(item: Item):
#     input_data = item.input_data
#
#     if input_data.strip().endswith("="):
#         base64_bytes = input_data.encode("utf-8")
#         decoded_bytes = base64.b64decode(base64_bytes)
#         decoded_string = decoded_bytes.decode("utf-8")
#         data = pd.read_csv(io.StringIO(decoded_string))
#
#     elif input_data.startswith("http"):
#         async with httpx.AsyncClient() as client:
#             response = await client.get(input_data)
#             if response.status_code == 200:
#                 with open(
#                     os.path.join(zkp_dir, os.path.basename(input_data)), "wb"
#                 ) as f:
#                     f.write(response.content)
#             else:
#                 return {"error": f"Failed to download {input_data}"}
#         data = pd.read_csv(io.BytesIO(response.content))
#     else:
#         data = pd.read_csv(input_data)
#
#     data = data.iloc[:1, :]
#
#     data_path = convert_model_data(test_df_set=data)
#     if not os.path.exists(data_path):
#         return {"files": []}
#
#     await inference_ekzl(data_path=data_path)
#
#     files_to_send = glob.glob(os.path.join(zkp_dir, "*"))
#     return {
#         "files": [
#             f"http://{HOST}:{PORT}/download/" + f.split("/")[-1]
#             for f in files_to_send
#             if f.split("/")[-1]
#             in ["test.vk", "test.pf", "kzg.srs", "settings.json"]
#         ]
#     }


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_location = glob.glob(os.path.join(zkp_dir, "*"))

    required_file = [f for f in file_location if filename in f]
    if len(required_file) != 0:
        required_file = required_file[0]
        return FileResponse(
            required_file, media_type="application/octet-stream", filename=filename
        )
    else:
        return {"err": "file not found"}


# @app.post("/verify")
# async def verify_files_url(urls=Body(...)):
#     for field, url in urls.items():
#         response = requests.get(url, stream=True)
#         if response.status_code == 200:
#             with open(os.path.join(zkp_dir, field), "wb") as f:
#                 f.write(response.content)
#         else:
#             return {"error": f"Failed to download {url}"}
#
#     result = await verify(
#         proof_path=os.path.join(zkp_dir, "test_pf.file"),
#         settings_path=os.path.join(zkp_dir, "settings_json.file"),
#         vk_path=os.path.join(zkp_dir, "test_vk.file"),
#         srs_path=os.path.join(zkp_dir, "kzg_srs.file"),
#     )
#
#     return {"result": result}


# @app.post("/verify")
# async def verify_files_url(urls=Body(...)):
#     async with httpx.AsyncClient() as client:
#         for field, url in urls.items():
#             response = await client.get(url)
#             if response.status_code == 200:
#                 with open(os.path.join(zkp_dir, os.path.basename(url)), "wb") as f:
#                     f.write(response.content)
#             else:
#                 return {"error": f"Failed to download {url}"}
#
#     result = await verify(
#         proof_path=os.path.join(zkp_dir, "test.pf"),
#         settings_path=os.path.join(zkp_dir, "settings.json"),
#         vk_path=os.path.join(zkp_dir, "test.vk"),
#         srs_path=os.path.join(zkp_dir, "kzg.srs"),
#     )
#
#     return {"result": result}


# @app.post("/verify")
# async def verify_files_url(inputs=Body(...)):
#     for field, input_data in inputs.items():
#         base64_bytes = input_data.encode("utf-8")
#         decoded_bytes = base64.b64decode(base64_bytes)
#         # decoded_string = decoded_bytes.decode("utf-8")
#         with open(os.path.join(zkp_dir, field), "wb") as f:
#             f.write(decoded_bytes)
#
#     result = await verify(
#         proof_path=os.path.join(zkp_dir, "test.pf"),
#         settings_path=os.path.join(zkp_dir, "settings.json"),
#         vk_path=os.path.join(zkp_dir, "test.vk"),
#         srs_path=os.path.join(zkp_dir, "kzg.srs"),
#     )
#
#     return {"result": result}
@app.post("/verify")
async def verify_files_url(inputs=Body(...)):
    result = []
    inputs_ = inputs.get("output")
    for input_data in inputs_:
        for field, file_item in input_data.items():
            base64_bytes = file_item.encode("utf-8")
            decoded_bytes = base64.b64decode(base64_bytes)
            # decoded_string = decoded_bytes.decode("utf-8")
            with open(os.path.join(zkp_dir, field), "wb") as f:
                f.write(decoded_bytes)

        r = await verify(
            proof_path=os.path.join(zkp_dir, "test.pf"),
            settings_path=os.path.join(zkp_dir, "settings.json"),
            vk_path=os.path.join(zkp_dir, "test.vk"),
            srs_path=os.path.join(zkp_dir, "kzg.srs"),
        )
        result.append(r)

    return {"result": result}


@app.post("/verify_path")
async def verify_files(
    test_vk: UploadFile = File(...),
    test_pf: UploadFile = File(...),
    kzg_srs: UploadFile = File(...),
    settings_json: UploadFile = File(...),
):
    with open(os.path.join(zkp_dir, test_vk.filename), "wb") as buffer:
        shutil.copyfileobj(test_vk.file, buffer)

    with open(os.path.join(zkp_dir, test_pf.filename), "wb") as buffer:
        shutil.copyfileobj(test_pf.file, buffer)

    with open(os.path.join(zkp_dir, kzg_srs.filename), "wb") as buffer:
        shutil.copyfileobj(kzg_srs.file, buffer)

    with open(os.path.join(zkp_dir, settings_json.filename), "wb") as buffer:
        shutil.copyfileobj(settings_json.file, buffer)

    result = await verify(
        proof_path=os.path.join(zkp_dir, test_pf.filename),
        settings_path=os.path.join(zkp_dir, settings_json.filename),
        vk_path=os.path.join(zkp_dir, test_vk.filename),
        srs_path=os.path.join(zkp_dir, kzg_srs.filename),
    )

    return {"result": result}
