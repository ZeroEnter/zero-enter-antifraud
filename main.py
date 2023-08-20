import os

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

    await inference_ekzl(data_path=input.file)

    files_to_send = glob.glob(os.path.join(zkp_dir, "*"))
    return {
        "file_urls": [
            f
            for f in files_to_send
            if f.split("/")[-1] in ["test.vk", "test.pf", "kzg.srs", "settings.json"]
        ]
    }


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_location = glob.glob(os.path.join(zkp_dir, "*"))
    required_file = [f for f in file_location if filename in f][0]
    return FileResponse(
        required_file, media_type="application/octet-stream", filename=filename
    )


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

    result = await verify(proof_path=test_pf.file,
                          settings_path=settings_json.file,
                          vk_path=test_vk.file,
                          srs_path=kzg_srs.file,
                          )

    return {"result": result}
