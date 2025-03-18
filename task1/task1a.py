# Experiment how to make M a learnable model parameter and using SGD to optimise this more flexible model.
# Report using printed messages, the optimised M value and the associated metric values, on the model prediction.
import torch
import torch.nn as nn
import torch.optim as optim
from math import comb, sqrt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task import get_exp_combinations, generate_sample, MyCrossEntropy, compute_accuracy, compute_true_labels

class FlexibleLogisticModel(nn.Module):
    def __init__(self, D, M_max):
        super(FlexibleLogisticModel, self).__init__()
        self.D = D
        self.M_max = M_max
        
        # Precompute number of features for each order m
        self.feature_counts = [int(comb(D + m - 1, m)) for m in range(M_max+1)]
        self.total_features = sum(self.feature_counts)
        
        # Weight vector for all polynomial features
        self.w = nn.Parameter(torch.randn(self.total_features))
        
        # Learnable scalar for each polynomial order
        init_alphas = torch.ones(M_max+1)
        for m in range(1, M_max+1):
            init_alphas[m] = 1.0 / (2**m)  # Prior favoring lower orders
        self.alphas = nn.Parameter(init_alphas)

    def forward(self, x):
        all_features = []
        feature_idx = 0
        
        # For each polynomial order
        for m in range(self.M_max + 1):
            # Get features for this order
            idxs = get_exp_combinations(self.D, m)
            order_features = []
            
            for idx in idxs:
                exponents = torch.tensor(idx, dtype=x.dtype, device=x.device)
                term = torch.prod(x ** exponents)
                order_features.append(term)
            
            # Stack features for this order and scale by alpha
            if order_features:
                order_features = torch.stack(order_features)
                scaled_features = self.alphas[m] * order_features
                all_features.extend(scaled_features.tolist())
                feature_idx += len(order_features)
        
        # Convert to tensor and compute logistic output
        phi = torch.tensor(all_features, dtype=x.dtype, device=x.device)
        f = torch.dot(self.w, phi)
        return torch.sigmoid(f)

    def get_effective_M(self):
        """Return the highest polynomial order with significant weight"""
        threshold = 0.1
        max_order = 0
        for m in range(self.M_max, -1, -1):
            if abs(self.alphas[m].item()) > threshold:
                max_order = m
                break
        return max_order

def experiment_with_learnable_M(x_train, t_train, x_test, t_test, W_true, M_true, D):
    """Train a model with learnable polynomial order and report results"""
    print("\n=== Experiment: Making M a Learnable Parameter ===")
    
    # Parameters
    M_max = 5
    model = FlexibleLogisticModel(D, M_max)
    loss_fn = MyCrossEntropy()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Training settings
    epochs = 200
    batch_size = 32
    N_train = len(x_train)
    
    for epoch in range(epochs):
        perm = torch.randperm(N_train) # Shuffle data
        x_shuffled = x_train[perm]
        t_shuffled = t_train[perm]
        
        # Minibatch training
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, N_train, batch_size):
            x_batch = x_shuffled[i:i+batch_size]
            t_batch = t_shuffled[i:i+batch_size]
            
            # Forward pass
            optimizer.zero_grad()
            y_preds = torch.stack([model(x) for x in x_batch])
            
            loss = loss_fn(y_preds, t_batch)
            total_loss += loss.item()
            num_batches += 1
            
            loss.backward()
            optimizer.step()
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
            print(f"Alpha weights: {model.alphas.data}")
    
    # Get the effective polynomial order
    effective_M = model.get_effective_M()
    
    # Compute metrics
    y_train_preds = torch.stack([model(x) for x in x_train])
    y_test_preds = torch.stack([model(x) for x in x_test])
    
    # Compute accuracy using true underlying labels
    true_train = compute_true_labels(x_train, W_true, M_true)
    true_test = compute_true_labels(x_test, W_true, M_true)
    
    train_acc = compute_accuracy(y_train_preds, true_train)
    test_acc = compute_accuracy(y_test_preds, true_test)
    
    # reporting the effective polynomial order and associated metric values 
    print("\nTraining complete!")
    print(f"Learned order weights (alphas): {model.alphas.data}")
    print(f"Effective polynomial order M: {effective_M}")
    print(f"Model prediction accuracy on training set: {train_acc * 100:.2f}%")
    print(f"Model prediction accuracy on test set: {test_acc * 100:.2f}%")

    return effective_M, train_acc, test_acc, model.alphas.data.tolist()

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    M_data = 2
    D = 5
    p = sum(comb(D + m - 1, m) for m in range(M_data + 1))
    
    # Generate underlying true weight vector
    W_list = [((-1)**(p-i)) * (sqrt(p-i) / p) for i in range(p)]
    W = torch.tensor(W_list, dtype=torch.float32)
    print("Underlying weight vector W:")
    print(W)
    
    # Generate training data
    N_train = 200
    train_samples = [generate_sample(W, M_data, D) for _ in range(N_train)]
    x_train = torch.stack([s[0] for s in train_samples])
    t_train = torch.tensor([s[1] for s in train_samples], dtype=torch.float32)
    
    # Generate test data
    N_test = 100
    test_samples = [generate_sample(W, M_data, D) for _ in range(N_test)]
    x_test = torch.stack([s[0] for s in test_samples])
    t_test = torch.tensor([s[1] for s in test_samples], dtype=torch.float32)
    
    # Run experiment with learnable M
    effective_M, train_acc, test_acc, alphas = experiment_with_learnable_M(
        x_train, t_train, x_test, t_test, W, M_data, D
    )