import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from task2.my_elm import MyExtremeLearningMachine


class MyEnsembleELM(nn.Module):
    """
    Ensemble of Extreme Learning Machine models for improved classification performance.

    Creates an ensemble of multiple independent ELM models and combines their outputs
    through soft voting (averaging probabilities) for final predictions.
    """

    def __init__(
        self,
        seed=42,
        n_models=5,
        num_feature_maps=32,
        std_dev=0.5,
        feature_size=None,
        kernel_size=3,
        pooling=False,
        reproduce=True,
    ):
        """
        Initialize the Ensemble ELM model.

        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            n_models (int, optional): Number of models in the ensemble. Defaults to 5.
            num_feature_maps (int, optional): Number of feature maps for each ELM model. Defaults to 32.
            std_dev (float, optional): Standard deviation for weight initialization. Defaults to 0.5.
            feature_size (int, optional): Size of feature vector. If None, calculated automatically. Defaults to None.
            kernel_size (int, optional): Size of convolution kernel. Defaults to 3.
            pooling (bool, optional): Whether to use pooling in the models. Defaults to False.
            reproduce (bool, optional): Whether to use deterministic initialization. Defaults to True.
        """
        super(MyEnsembleELM, self).__init__()

        # wranings
        if n_models < 3 or n_models > 10:
            print(
                f"Warning: n_models={n_models} is outside the recommended range [3, 10]"
            )

        if num_feature_maps < 16 or num_feature_maps > 128:
            print(
                f"Warning: num_feature_maps={num_feature_maps} is outside the recommended range [16, 128]"
            )

        if std_dev < 0.01 or std_dev > 1.0:
            print(
                f"Warning: std_dev={std_dev} is outside the recommended range [0.01, 1.0]"
            )

        # set seed for reproducibility
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.models = nn.ModuleList()
        self.n_models = n_models

        for i in range(n_models):
            # set seed for each model, if reproduce=False, then random seed
            if reproduce:
                model_seed = seed + i
                torch.manual_seed(model_seed)
                np.random.seed(model_seed)
            else:
                model_seed = np.random.randint(0, 10000)
                torch.manual_seed(model_seed)
                np.random.seed(model_seed)

            # manual calculation for CIFAR-10 images (32x32) with 3x3 kernel
            if feature_size is None:
                h_out = 32 - kernel_size + 1
                w_out = 32 - kernel_size + 1
                if pooling:
                    h_out = h_out // 2
                    w_out = w_out // 2
                feature_size = num_feature_maps * h_out * w_out

            model = MyExtremeLearningMachine(
                num_feature_maps=num_feature_maps,
                num_classes=10,
                std_dev=std_dev,
                feature_size=feature_size,
                kernel_size=kernel_size,
                pooling=pooling,
            )
            self.models.append(model)

    def to(self, device):
        """
        Move all models in the ensemble to the specified device.

        Args:
            device (str or torch.device): Device to move models to (e.g., 'cpu', 'cuda')

        Returns:
            MyEnsembleELM: Self reference for chaining operations
        """
        for model in self.models:
            model.to(device)
        return self

    def train(self, train_loader, fit_function, **kwargs):
        """
        Train all models in the ensemble.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training data
            fit_function (callable): Function used to train each model
            **kwargs: Additional arguments to pass to the fit_function

        Returns:
            None
        """
        for i, model in enumerate(self.models):
            print(f"Training model {i + 1}/{self.n_models}")
            fit_function(model, train_loader, **kwargs)

    def pred(self, x):
        """
        Combine predictions from all models in the ensemble (soft voting).

        Args:
            x (torch.Tensor): Input data of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Ensemble prediction probabilities of shape (batch_size, num_classes)
        """
        for model in self.models:
            model.eval()
        # get preds from each model
        with torch.no_grad():
            # new tensor to keep track of combined preds
            ensemble_output = torch.zeros((x.size(0), 10), device=x.device)

            # sum preds from each model
            for model in self.models:
                outputs = model(x)
                probs = F.softmax(outputs, dim=1)
                ensemble_output += probs

            # avg the preds
            ensemble_output /= len(self.models)

            return ensemble_output

    def eval(self, test_loader=None, device="cpu"):
        """
        Evaluate the ensemble on test data.

        Args:
            test_loader (torch.utils.data.DataLoader, optional): DataLoader for test data.
                                                            If None, just sets models to eval mode. Defaults to None.
            device (str, optional): Device to run evaluation on. Defaults to "cpu".

        Returns:
            float or None: Accuracy on test data if test_loader is provided, None otherwise
        """
        if test_loader is None:
            for model in self.models:
                model.eval()
            return

        self.to(device)
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.pred(inputs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = correct / total
        return accuracy

    def forward(self, x):
        """
        Forward pass for the ensemble (wrapper for pred method).

        Args:
            x (torch.Tensor): Input data of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Ensemble prediction probabilities of shape (batch_size, num_classes)
        """
        return self.pred(x)
