"""Train a small linear model using the gguf_train torch-compatible API."""
import torch
import torch.nn as nn
import torch.optim as optim


def main():
    torch.manual_seed(5)
    x = torch.randn(64, 4)
    target = x @ torch.tensor([[2.0], [-1.0], [0.5], [3.0]])
    model = nn.Linear(4, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    for _ in range(150):
        optimizer.zero_grad()
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()
        optimizer.step()
    print("final loss:", loss.item())
    print("weight:", model.weight.tolist())


if __name__ == "__main__":
    main()
