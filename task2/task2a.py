import torch
import torch.nn.functional as F
import time

# TODO: Add new metrics, and consider adding loss as well for viz 
def fit_elm_ls(model, train_loader, lambda_reg = 0.001, device="cpu", num_epochs=10):
    """
    Using a variant of the least squares algorithm to train the ELM - ridge
    device should always be CPU due to the library constrains but just in case cuda is used
    """
    start_time = time.time()
    model = model.to(device)
    model.eval()

    # feature and targets
    all_features = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # get the features from the model
            features = model.conv(inputs)
            features = F.relu(features)
            features = features.view(features.size(0), -1)

            all_features.append(features)
            all_targets.append(targets)
        
        # concat all batches
        x = torch.cat(all_features, dim=0)
        y = torch.cat(all_targets, dim=0)

        # one hot encoding of the targets
        y_onehot = torch.zeros(y.size(0), model.num_classes, device=device)
        y_onehot.scatter_(1, y.unsqueeze(1), 1)

        # solve using ridge regression
        XTX = torch.mm(x.t(), x)
        reg_term = lambda_reg * torch.eye(XTX.size(0), device=device)
        XTy = torch.matmul(x.t(), y_onehot)

        # solve (X^T X + λI)W = X^T y
        W = torch.linalg.solve(XTX + reg_term, XTy)

        # update weights
        with torch.no_grad():
            model.fc.weight.copy_(W.t())

            if model.fc.bias is not None:
                # compute bias
                predicted = torch.matmul(x, W)
                bias = torch.mean(y_onehot - predicted, dim=0)
                model.fc.bias.copy_(bias)
        
        # eval on training
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        train_acc = correct / total
        print(f"Training completed with least-squares solver. Accuracy: {train_acc:.4f}")
        
        return {"train_acc": [train_acc]}
