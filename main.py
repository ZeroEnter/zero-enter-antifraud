import click

# Define the network client
from xrpl.clients import JsonRpcClient
from xrpl.models import Payment
from xrpl.transaction import submit_and_wait, sign_and_submit
from xrpl.utils import xrp_to_drops
from xrpl.wallet import Wallet

from pyze.memo import generate_verification_memos

JSON_RPC_URL = "http://127.0.0.1:5005"
client = JsonRpcClient(JSON_RPC_URL)


def get_ml_proof_memos():
    return generate_verification_memos()


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
