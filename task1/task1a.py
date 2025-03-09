# Experiment how to make M a learnable model parameter and using SGD to optimise this more flexible model.
# Report using printed messages, the optimised M value and the associated metric values, on the model prediction.
import torch
import torch.nn as nn
import torch.optim as optim
from math import comb, sqrt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task import get_exp_combinations, generate_sample

class FlexibleLogisticModel(nn.Module):
    def __init__(self, D, M_max):
        super(FlexibleLogisticModel, self).__init__()
        self.D = D
        self.M_max = M_max
        # Precompute number of features for each order m.
        self.feature_counts = [int(comb(D + m - 1, m)) for m in range(M_max+1)]
        self.total_features = sum(self.feature_counts)
        # Weight vector for all polynomial features.
        self.w = nn.Parameter(torch.randn(self.total_features))
        # Learnable scalar for each polynomial order.
        self.alphas = nn.Parameter(torch.ones(M_max+1))
    
    def forward(self, x):
        features = []
        for m in range(self.M_max+1):
            idxs = get_exp_combinations(self.D, m)
            phi_m = []
            for idx in idxs:
                exponents = torch.tensor(idx, dtype=x.dtype, device=x.device)
                phi_m.append(torch.prod(x ** exponents))
            phi_m = torch.stack(phi_m)
            # Scale features of order m by alpha_m
            features.append(self.alphas[m] * phi_m)

        phi = torch.cat(features) 
        f = torch.dot(self.w, phi)
        return torch.sigmoid(f)

# Experiment: Learn flexible polynomial order.
# Parameters for data generation.
M_data = 2
D = 5
p = sum(comb(D + m - 1, m) for m in range(M_data + 1))  # p = 1 + 5 + 15 = 21

# Generate underlying true weight vector.
W_list = [((-1)**(p - i)) * (sqrt(p - i) / p) for i in range(p)]
W = torch.tensor(W_list, dtype=torch.float32)
print("Underlying weight vector W:")
print(W)

# Generate 200 training samples.
N_train = 200
train_samples = [generate_sample(W, M_data, D) for _ in range(N_train)]
x_train = torch.stack([s[0] for s in train_samples])
t_train = torch.tensor([s[1] for s in train_samples], dtype=torch.float32)

# Generate 100 test samples.
N_test = 100
test_samples = [generate_sample(W, M_data, D) for _ in range(N_test)]
x_test = torch.stack([s[0] for s in test_samples])
t_test = torch.tensor([s[1] for s in test_samples], dtype=torch.float32)

model = FlexibleLogisticModel(D, M_data)
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    losses = []
    for i in range(x_train.size(0)):
        y_pred = model(x_train[i])
        y_pred = torch.clamp(y_pred, 1e-7, 1 - 1e-7)
        loss = - (t_train[i] * torch.log(y_pred) + (1-t_train[i]) * torch.log(1-y_pred))
        losses.append(loss)
    loss_epoch = torch.stack(losses).mean()
    loss_epoch.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss_epoch.item()}")

print("Learned order weights (alphas):", model.alphas.data)