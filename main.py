import base64
import glob
import io
import os
import shutil

import httpx
import pandas as pd
import requests
from ezkl import ezkl
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ezkl_inference import inference_ekzl, convert_model_data

HOST = os.getenv("HOST", "localhost")
PORT = os.getenv("PORT", 8000)

zkp_dir = "ezkl_inference/data_zkp"
os.makedirs(zkp_dir, exist_ok=True)

app = FastAPI()


class Item(BaseModel):
    input_data: str


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

    if input_data.strip().endswith("="):
        base64_bytes = input_data.encode("utf-8")
        decoded_bytes = base64.b64decode(base64_bytes)
        decoded_string = decoded_bytes.decode("utf-8")
        data = pd.read_csv(io.StringIO(decoded_string))

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
        data = pd.read_csv(io.BytesIO(response.content))
    else:
        data = pd.read_csv(input_data)

    data = data.iloc[:1, :]

    data_path = convert_model_data(test_df_set=data)
    if not os.path.exists(data_path):
        return {"files": []}

    await inference_ekzl(data_path=data_path)

    files_to_send = glob.glob(os.path.join(zkp_dir, "*"))
    return {
        "files": [
            f"http://{HOST}:{PORT}/download/" + f.split("/")[-1]
            for f in files_to_send
            if f.split("/")[-1]
            in ["test.vk", "test.pf", "kzg.srs", "settings.json"]
        ]
    }


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


@app.post("/verify")
async def verify_files_url(urls=Body(...)):
    async with httpx.AsyncClient() as client:
        for field, url in urls.items():
            response = await client.get(url)
            if response.status_code == 200:
                with open(os.path.join(zkp_dir, os.path.basename(url)), "wb") as f:
                    f.write(response.content)
            else:
                return {"error": f"Failed to download {url}"}

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
