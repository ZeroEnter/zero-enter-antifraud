import torch
import os
from models import SimpleAntiFraudGNN, SimpleKYC


dir2save_model = "weights"
os.makedirs(dir2save_model, exist_ok=True)

model = SimpleKYC()
model = model.to(torch.device("cpu"))
model.train()
model.eval()

path2save_weights = os.path.join(
    dir2save_model, f"model_{model.__class__.__name__}.pth"
)
torch.save(model.state_dict(), path2save_weights)
print(f"model saved_to: {path2save_weights}")