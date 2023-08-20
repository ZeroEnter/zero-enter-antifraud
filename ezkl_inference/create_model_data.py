import json
import os
import shutil

import torch

from inference_model_ezkl import preproc_data_features
from models import SimpleAntiFraudGNN

zkp_dir = "ezkl_inference/data_zkp"
os.makedirs(zkp_dir, exist_ok=True)


def create_model_data(
    data_path=os.path.join(zkp_dir, "input.json"),
    model_path=os.path.join(zkp_dir, "network.onnx"),
):
    device = torch.device("cpu")
    features = preproc_data_features()

    model = SimpleAntiFraudGNN()
    dir2save_model = "weights"
    path2save_weights = os.path.join(
        dir2save_model, f"model_{model.__class__.__name__}.pth"
    )
    model.load_state_dict(
        torch.load(path2save_weights, map_location=device)
    )  # Choose whatever GPU device number you want
    model.to(device)

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
    return True


if __name__ == "__main__":
    create_model_data()
