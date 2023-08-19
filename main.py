import torch

from ezkl_inference import inference_ekzl, preproc_data_features


def main():
    d = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(d) if torch.cuda.is_available() else torch.device(d)

    features = preproc_data_features()
    output = inference_ekzl(features=features, device=device)

    return output


if __name__ == "__main__":
    main()
