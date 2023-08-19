import os

import torch
from torch import nn
from tqdm import tqdm


def simple_train_loop(
    num_epochs: 201,
    model,
    features,
    targets,
    dir2save_model: str = "weights",
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device)
    model = model.to(device)
    features = features.to(device)
    targets = targets.to(device)

    model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # Train the model
    for i in tqdm(range(num_epochs)):
        # Forward pass
        output = model(features)
        # Compute the loss
        loss = criterion(output, targets)
        if i % 20 == 0:
            print(f"Epoch: {i}, Loss: {loss.item()}")
        # Zero the gradients
        optimizer.zero_grad()
        # Perform backpropagation
        loss.backward()
        # Update the parameters
        optimizer.step()

    os.makedirs(dir2save_model, exist_ok=True)

    model.eval()
    model = model.to(torch.device("cpu"))

    path2save_weights = os.path.join(
        dir2save_model, f"model_{model.__class__.__name__}.pth"
    )
    torch.save(model.state_dict(), path2save_weights)
    print(f"model saved_to: {path2save_weights}")
    return model
