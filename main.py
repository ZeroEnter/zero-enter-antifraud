import json
import os
import requests
from ezkl import ezkl

from ezkl_inference import inference_ekzl
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import FileResponse
import shutil
import asyncio
import glob
import httpx

HOST = os.getenv("HOST", "localhost")
PORT = os.getenv("PORT", 8000)

zkp_dir = "ezkl_inference/data_zkp"
os.makedirs(zkp_dir, exist_ok=True)

app = FastAPI()


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
async def create_inference(input_data=Body(...)):
    # input_data = json.loads(input)  # assuming that the input is JSON data as a string
    filename = "input.json"
    with open(os.path.join(zkp_dir, filename), "w") as file:
        json.dump(input_data, file)

    await inference_ekzl(data_path=os.path.join(zkp_dir, filename))

    files_to_send = glob.glob(os.path.join(zkp_dir, "*"))
    return {
        "files": [
            f"http://{HOST}:{PORT}/download/" + f.split("/")[-1]
            for f in files_to_send
            if f.split("/")[-1]
            in ["test_ver.vk", "test_ver.pf", "kzg.srs", "settings.json"]
        ]
    }


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_location = glob.glob(os.path.join(zkp_dir, "*"))
    required_file = [f for f in file_location if filename in f][0]
    return FileResponse(
        required_file, media_type="application/octet-stream", filename=filename
    )


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
                with open(os.path.join(zkp_dir, field), "wb") as f:
                    f.write(response.content)
            else:
                return {"error": f"Failed to download {url}"}

    result = await verify(
        proof_path=os.path.join(zkp_dir, "test_pf.file"),
        settings_path=os.path.join(zkp_dir, "settings_json.file"),
        vk_path=os.path.join(zkp_dir, "test_vk.file"),
        srs_path=os.path.join(zkp_dir, "kzg_srs.file"),
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
