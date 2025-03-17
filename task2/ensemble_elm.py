import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from task2.my_elm import MyExtremeLearningMachine

class MyEnsembleELM(nn.Module):
    def __init__(self, seed=42, n_models=5, num_feature_maps=32, std_dev=0.5, feature_size=None, kernel_size=3, pooling=False, reproduce=True):
        """CIFAR-10 default values for kernel_size, num_feature_maps, std_dev, pooling to speed up training"""
        super(MyEnsembleELM, self).__init__()
        
        # wranings
        if n_models < 3 or n_models > 10:
            print(f"Warning: n_models={n_models} is outside the recommended range [3, 10]")
    
        if num_feature_maps < 16 or num_feature_maps > 128:
            print(f"Warning: num_feature_maps={num_feature_maps} is outside the recommended range [16, 128]")
            
        if std_dev < 0.01 or std_dev > 1.0:
            print(f"Warning: std_dev={std_dev} is outside the recommended range [0.01, 1.0]")

        # set seed for reproducibility
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.models = nn.ModuleList()
        self.n_models = n_models

        for i in range(n_models):
            # set seed
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
                pooling=pooling
            )
            self.models.append(model)

    def to(self, device):
        """Move all models to specified device"""
        for model in self.models:
            model.to(device)
        return self

    def train(self, train_loader, fit_function, **kwargs):
        """Train all models in the ensemble"""
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{self.n_models}")
            fit_function(model, train_loader, **kwargs)

    def pred(self, x):
        """Combine predictions from all models (soft voting)"""
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

    def eval(self, test_loader=None, device='cpu'):
        if test_loader is None:
            for model in self.models:
                model.eval()
            return
        # change device maybe?
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
        """Wrapper forward for pred in the ensemble"""
        return self.pred(x)