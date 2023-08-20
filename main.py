import os
import requests
from ezkl import ezkl

from ezkl_inference import inference_ekzl
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import shutil
import asyncio
import glob

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
async def create_inference(input: UploadFile = File(...)):
    with open(os.path.join(zkp_dir, input.filename), "wb") as buffer:
        shutil.copyfileobj(input.file, buffer)

    await inference_ekzl(data_path=os.path.join(zkp_dir, input.filename))

    files_to_send = glob.glob(os.path.join(zkp_dir, "*"))
    return {
        "files": [
            f.split("/")[-1]
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


@app.post("/verify_url")
async def verify_files_url(
    test_vk_url: str,
    test_pf_url: str,
    kzg_srs_url: str,
    settings_json_url: str,
):
    for url, filename in [
        (test_vk_url, "test_vk.file"),
        (test_pf_url, "test_pf.file"),
        (kzg_srs_url, "kzg_srs.file"),
        (settings_json_url, "settings_json.file"),
    ]:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(os.path.join(zkp_dir, filename), "wb") as f:
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


@app.post("/verify")
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