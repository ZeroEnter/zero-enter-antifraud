import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm


def simple_train_loop(
    num_epochs: 201,
    model,
    train_X,
    test_X,
    train_y,
    test_y,
    dir2save_model: str = "weights",
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device)
    model = model.to(device)

    train_X = train_X.to(device)
    test_X = test_X.to(device)
    train_y = train_y.to(device)
    test_y = test_y.to(device)

    model.train()

    loss_list = np.zeros((num_epochs,))
    accuracy_list = np.zeros((num_epochs,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # we use tqdm for nice loading bars
    for epoch in tqdm(range(num_epochs)):
        # To train, we get a prediction from the current network
        predicted_y = model(train_X)

        # Compute the loss to see how bad or good we are doing
        loss = loss_fn(predicted_y, train_y)

        # Append the loss to keep track of our performance
        loss_list[epoch] = loss.item()

        # Afterwards, we will need to zero the gradients to reset
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate the accuracy, call torch.no_grad() to prevent updating gradients
        # while calculating accuracy
        with torch.no_grad():
            y_pred = model(test_X)
            correct = (torch.argmax(y_pred, dim=1) == test_y).type(torch.FloatTensor)
            accuracy_list[epoch] = correct.mean()

    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(2, figsize=(12, 6), sharex=True)

    ax1.plot(accuracy_list)
    ax1.set_ylabel("Accuracy")
    ax2.plot(loss_list)
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("epochs")
    fig.savefig('docs/img_plot_metrics.png')

    os.makedirs(dir2save_model, exist_ok=True)

    model.eval()
    model = model.to(torch.device("cpu"))

    path2save_weights = os.path.join(
        dir2save_model, f"model_{model.__class__.__name__}.pth"
    )
    torch.save(model.state_dict(), path2save_weights)
    print(f"model saved_to: {path2save_weights}")
    return model
