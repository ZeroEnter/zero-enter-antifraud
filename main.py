import base64
import binascii
import glob
import io
import json
import os
import shutil

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
async def create_inference(item: Item):
    input_data = item.input_data
    type_model = item.type_model
    link_onnx = item.link_onnx
    model_path = os.path.join(zkp_dir, f"network_{type_model}.onnx")
    data_path = os.path.join(zkp_dir, f"input_{type_model}.json")

    async with httpx.AsyncClient() as client:
        response = await client.get(link_onnx)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
        else:
            return {"error": f"Failed to download {link_onnx}"}

    if type_model == "anti_fraud":
        base64_bytes = input_data.encode("utf-8")
        decoded_bytes = base64.b64decode(base64_bytes)
        decoded_string = decoded_bytes.decode("utf-8")
        data = pd.read_csv(io.StringIO(decoded_string))

        data = data.iloc[:1, :]

        data_path = convert_model_data(
            test_df_set=data, model_path=model_path, data_path=data_path
        )
        if not os.path.exists(data_path):
            return {"files": {"proof": None, "vk": None}}

    elif type_model == "simple_kyc":
        device = torch.device("cpu")

        model = SimpleKYC()
        # dir2save_model = "weights"
        # path2save_weights = os.path.join(
        #     dir2save_model, f"model_{model.__class__.__name__}.pth"
        # )
        # model.load_state_dict(
        #     torch.load(path2save_weights, map_location=device)
        # )  # Choose whatever GPU device number you want
        model.to(device)

        features = json.loads(input_data)
        te_x = torch.Tensor(features).float()
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

        data_array = features.detach().numpy().tolist()

        data = dict(input_data=data_array)

        # Serialize data into file:
        json.dump(data, open(data_path, "w"))
        return data_path
    else:
        return {"files": {"proof": None, "vk": None}}

    await inference_ekzl(data_path=data_path, model_path=model_path)

    with open(os.path.join(zkp_dir, "test.pf"), "r") as f:
        pf = json.load(f)

    with open(os.path.join(zkp_dir, "test.vk"), "rb") as f:
        vk = string_to_hex(f.read())

    return {"files": {"proof": pf["proof"], "vk": vk}}
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


@app.post("/verify")
async def verify_files_url(inputs=Body(...)):
    for field, input_data in inputs.items():
        if input_data.strip().endswith("="):
            base64_bytes = input_data.encode("utf-8")
            decoded_bytes = base64.b64decode(base64_bytes)
            # decoded_string = decoded_bytes.decode("utf-8")
            with open(os.path.join(zkp_dir, field), "wb") as f:
                f.write(decoded_bytes)

        elif input_data.startswith("http"):
            async with httpx.AsyncClient() as client:
                response = await client.get(input_data)
                if response.status_code == 200:
                    with open(
                        os.path.join(zkp_dir, os.path.basename(input_data)), "wb"
                    ) as f:
                        f.write(response.content)
                else:
                    return {"error": f"Failed to download {input_data}"}

    result = await verify(
        proof_path=os.path.join(zkp_dir, "test.pf"),
        settings_path=os.path.join(zkp_dir, "settings.json"),
        vk_path=os.path.join(zkp_dir, "test.vk"),
        srs_path=os.path.join(zkp_dir, "kzg.srs"),
    )

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
