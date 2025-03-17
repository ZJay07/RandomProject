import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from task2.metrics import evaluate_metrics


class MyExtremeLearningMachine(nn.Module):
    def __init__(
        self,
        num_feature_maps,
        num_classes,
        std_dev,
        feature_size,
        kernel_size=3,
        input_channels=3,
        pooling=False,
    ):
        """
        Init for fixed weights, hyperparameter to indicate the size of the hidden convolutional layer
        Feature maps to produce the multiclass prob vector suitable for image class
        Pooling to make training faster
        """
        super(MyExtremeLearningMachine, self).__init__()
        # one convo layer non trainable weights
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=num_feature_maps,
            kernel_size=kernel_size,
        )
        self.initialise_fixed_layers(std_dev)
        # Freeze the convolutional layer (make it non-trainable)
        for param in self.conv.parameters():
            param.requires_grad = False
        # non trainable fixed weights
        # fully connected layer with trainable weights
        self.fc = nn.Linear(feature_size, num_classes)
        self.num_classes = num_classes
        self.pooling = pooling

    def initialise_fixed_layers(self, std):
        """
        Function that init all the fixed weights in the convolution kernels
        Random sampling from a Gaussian distribution with zero mean and a sd
        """
        nn.init.normal_(self.conv.weight, mean=0.0, std=std)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        if self.pooling:
            x = F.avg_pool2d(x, 2)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x


def fit_elm_sgd(
    model, train_loader, test_loader, lr=0.01, device="cpu", num_epochs=10, eval_every=1
):
    """
    fit_elm_sgd with key metrics
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(trainable_params, lr=lr)

    statistics = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "test_f1": [],
        "epochs": [],
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        epoch_loss = train_loss / total
        epoch_acc = correct / total
        statistics["train_loss"].append(epoch_loss)
        statistics["train_acc"].append(epoch_acc)

        # eval with metrics
        if (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1:
            metrics = evaluate_metrics(model, test_loader, device)
            statistics["test_acc"].append(metrics["accuracy"])
            statistics["test_f1"].append(metrics["macro_f1"])
            statistics["epochs"].append(epoch + 1)

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
                f"Test Acc: {metrics['accuracy']:.2f}%, Test F1: {metrics['macro_f1']:.2f}%"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}"
            )

    final_metrics = evaluate_metrics(model, test_loader, device)

    return statistics, final_metrics
