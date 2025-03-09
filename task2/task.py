import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from task2.mix_up import MyMixUP


class MyExtremeLearningMachine(nn.Module):
    def __init__(
        self,
        num_feature_maps,
        num_classes,
        std_dev,
        feature_size,
        kernel_size=3,
        input_channels=3,
    ):
        """
        Init for fixed weights, hyperparameter to indicate the size of the hidden convolutional layer
        Feature maps to produce the multiclass prob vector suitable for image class
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
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x


def fit_elm_sgd(model, train_loader, lr=0.01, device="cpu", num_epochs=10):
    """
    Using a variant of the stochastic gradient descent algorithm to train the ELM
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        trainable_params,
        lr=lr,
    )
    statistic_dict = {
        "train_loss": [],
        "train_acc": [],
    }
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for i, (inputs, targets) in enumerate(train_loader, 0):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # zero the parameter gradients
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        epoch_loss = train_loss / train_total
        epoch_acc = train_correct / train_total
        statistic_dict["train_loss"].append(epoch_loss)
        statistic_dict["train_acc"].append(epoch_acc)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}"
        )
    return statistic_dict
