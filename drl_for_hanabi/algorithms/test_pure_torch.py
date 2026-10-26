import torch
net = torch.nn.Linear(10, 2)
x = torch.randn(10)
out = net(x)
loss = out.sum()
print('forward pass done:', loss.item())
loss.backward()
print('backward pass OK - pure torch, no Hanabi code at all')
