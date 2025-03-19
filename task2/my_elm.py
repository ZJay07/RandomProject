import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from task2.metrics_and_visualisation import evaluate_metrics


class MyExtremeLearningMachine(nn.Module):
    """
    Implementation of Extreme Learning Machine (ELM) for image classification.
    
    An ELM consists of a single convolutional layer with fixed (non-trainable) weights
    followed by a fully connected layer with trainable weights. The fixed weights are
    initialized randomly and remain unchanged during training.
    """
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
        Initialize the Extreme Learning Machine model.
        
        Args:
            num_feature_maps (int): Number of feature maps in the convolutional layer
            num_classes (int): Number of output classes
            std_dev (float): Standard deviation for weight initialization
            feature_size (int): Size of the flattened feature vector after convolution
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
            input_channels (int, optional): Number of input channels. Defaults to 3.
            pooling (bool, optional): Whether to use average pooling. Defaults to False.
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
        Initialize fixed weights in the convolutional layer.
        
        Weights are randomly sampled from a Gaussian distribution with zero mean
        and the specified standard deviation.
        
        Args:
            std (float): Standard deviation for the Gaussian initialization
            
        Returns:
            None
        """
        nn.init.normal_(self.conv.weight, mean=0.0, std=std)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
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
    Train an Extreme Learning Machine model using Stochastic Gradient Descent.
    
    Only the fully connected layer weights are updated during training.
    The model is evaluated on the test set periodically to track performance.
    
    Args:
        model (MyExtremeLearningMachine): The ELM model to train
        train_loader (torch.utils.data.DataLoader): DataLoader for training data
        test_loader (torch.utils.data.DataLoader): DataLoader for test data
        lr (float, optional): Learning rate for SGD. Defaults to 0.01.
        device (str, optional): Device to train on ('cpu' or 'cuda'). Defaults to "cpu".
        num_epochs (int, optional): Number of training epochs. Defaults to 10.
        eval_every (int, optional): Frequency of evaluation on test set. Defaults to 1.
        
    Returns:
        tuple: A tuple containing:
            - statistics (dict): Dictionary with training and testing metrics over time
            - final_metrics (dict): Dictionary with final evaluation metrics
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
