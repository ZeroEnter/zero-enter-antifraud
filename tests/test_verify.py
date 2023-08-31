import requests
import base64
import os


def to_b64(path_to_file):

    with open(path_to_file, "rb") as f:
        file_content = f.read()

    base64_encoded = base64.b64encode(file_content).decode("utf-8")
    return base64_encoded


def test():
    zkp_dir = "ezkl_inference/data_zkp"
    url = "http://localhost:8000/verify"

    type_model = "anti_fraud"
    vk_path = os.path.join(zkp_dir, f"test_{type_model}.vk")
    settings_path = os.path.join(zkp_dir, f"settings_{type_model}.json")
    srs_path = os.path.join(zkp_dir, f"kzg_{type_model}.srs")
    proof_path = os.path.join(zkp_dir, f"test_{type_model}.pf")

    headers = {"Content-Type": "application/json"}
    data = {"input_data": [{
        "test.vk": to_b64(vk_path),
        "test.pf": to_b64(proof_path),
        "kzg.srs": to_b64(srs_path),
        "settings.json": to_b64(settings_path),
    }]}

    response = requests.post(url, headers=headers, json=data)
    print(response.content)


if __name__ == "__main__":
    test()