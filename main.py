import binascii
import os

import click
from ezkl import ezkl

# Define the network client
from xrpl.clients import JsonRpcClient
from xrpl.models import Payment, Memo
from xrpl.transaction import submit_and_wait, sign_and_submit
from xrpl.utils import xrp_to_drops
from xrpl.wallet import Wallet

JSON_RPC_URL = "http://127.0.0.1:5005"
client = JsonRpcClient(JSON_RPC_URL)

zkp_dir = "ezkl_inference/data_zkp"
os.makedirs(zkp_dir, exist_ok=True)


def string_to_hex(s):
    if isinstance(s, str):
        return binascii.hexlify(s.encode()).decode()

    return binascii.hexlify(s).decode()


def get_ml_proof_memos(
    model_path=os.path.join(zkp_dir, "network.onnx"),
    data_path=os.path.join(zkp_dir, "input.json"),
):
    compiled_model_path = os.path.join(zkp_dir, "network.compiled")
    pk_path = os.path.join(zkp_dir, "test.pk")
    vk_path = os.path.join(zkp_dir, "test.vk")
    settings_path = os.path.join(zkp_dir, "settings.json")
    srs_path = os.path.join(zkp_dir, "kzg.srs")
    witness_path = os.path.join(zkp_dir, "witness.json")
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

    return [
        Memo.from_dict(
            {
                "memo_data": res_proof["proof"],
                "memo_type": string_to_hex("type/model.proof"),
            }
        ),
    ]


@click.command()
@click.option(
    "--admin_seed", default="snoPBrXtMeMyMHUVTgbuqAfg1SUTb", help="Admin seed"
)  # seed 'sEdTE5SpNXuPc6h3oyGKaWfQXkyssPS'
@click.option(
    "--client_seed", default="sawmDjPis6h6AS9g5XYmviJ6N6EKu", help="Client seed"
)
def send_payment(admin_seed, client_seed):
    client_account = Wallet.from_seed(client_seed)
    admin_account = Wallet.from_seed(admin_seed)

    my_tx_payment = Payment(
        account=client_account.classic_address,
        amount=xrp_to_drops(22),
        destination=admin_account.classic_address,
        # memos=get_ml_proof_memos()
    )

    tx_response = sign_and_submit(my_tx_payment, client, client_account)
    print(tx_response)


if __name__ == "__main__":
    send_payment()
