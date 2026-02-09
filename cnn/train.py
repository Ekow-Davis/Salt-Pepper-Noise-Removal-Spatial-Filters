import torch
from torch.utils.data import DataLoader
from model import SimpleDenoiser
from dataset import NoisyImageDataset

dataset = NoisyImageDataset("data/clean", "data/noisy")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = SimpleDenoiser()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

for epoch in range(20):
    for noisy, clean in loader:
        output = model(noisy)
        loss = loss_fn(output, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "denoiser.pth")
