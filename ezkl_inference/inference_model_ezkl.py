import asyncio

from ezkl import ezkl
import os
from ezkl_inference.convert_model_data import *

zkp_dir = "ezkl_inference/data_zkp"
os.makedirs(zkp_dir, exist_ok=True)


async def inference_ekzl(data_path=os.path.join(zkp_dir, "input.json"), model_path=os.path.join(zkp_dir, "network.onnx"), type_model: str = ""):
    compiled_model_path = os.path.join(zkp_dir, f"network_{type_model}.compiled")
    pk_path = os.path.join(zkp_dir, f"test_{type_model}.pk")
    vk_path = os.path.join(zkp_dir, f"test_{type_model}.vk")
    settings_path = os.path.join(zkp_dir, f"settings_{type_model}.json")
    srs_path = os.path.join(zkp_dir, f"kzg_{type_model}.srs")
    witness_path = os.path.join(zkp_dir, f"witness_{type_model}.json")
    proof_path = os.path.join(zkp_dir, f"test_{type_model}.pf")

    run_args = ezkl.PyRunArgs()
    run_args.input_visibility = "private"
    run_args.param_visibility = "private"
    run_args.output_visibility = "public"

    res = ezkl.gen_settings(model_path, settings_path, py_run_args=run_args)
    assert res == True

    res = await ezkl.calibrate_settings(
        data_path, model_path, settings_path, "resources"
    )  # Optimize for resources

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

    # res = ezkl.verify(
    #     proof_path,
    #     settings_path,
    #     vk_path,
    #     srs_path,
    # )
    #
    # assert res == True
    # print("verified")
    return True


if __name__ == "__main__":
    # create_model_data()
    # inference_ekzl()
    asyncio.run(inference_ekzl())
