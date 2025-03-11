"""
A random guess in multiclass classification means assigning labels by pure chance. With K equally probable classes, the expected accuracy is 1/K (10% for CIFAR-10). To test if a model performs better than random guessing, 
compare its accuracy to this baseline using statistical tests like chi-squared or binomial tests. 
Calculate if the observed accuracy significantly exceeds the expected random accuracy. 
For imbalanced datasets, the random baseline becomes the frequency of the most common class. 
Random guessing represents the minimum performance threshold any useful classifier must exceed.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageDraw
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task2.my_elm import MyExtremeLearningMachine, fit_elm_sgd

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# CIFAR-10 Data Loading
def load_cifar10(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Function to evaluate model
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# Hyperparameter Experimentation
def experiment_hyperparameters():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader = load_cifar10(batch_size=128)
    
    # Hyperparameter grid
    feature_maps_options = [16, 32, 64, 128]
    std_dev_options = [0.01, 0.1, 0.5, 1.0]
    kernel_size_options = [3, 5]
    lr_options = [0.001, 0.01, 0.1]
    epochs_options = [5, 10, 20]
    
    # Results tracking
    results = []
    
    # Grid search
    for num_feature_maps in feature_maps_options:
        for std_dev in std_dev_options:
            for kernel_size in kernel_size_options:
                # Calculate feature size based on kernel size
                h_out = 32 - kernel_size + 1  # CIFAR images are 32x32
                w_out = 32 - kernel_size + 1
                feature_size = num_feature_maps * h_out * w_out
                
                for lr in lr_options:
                    for num_epochs in epochs_options:
                        print(f"\nTesting: feature_maps={num_feature_maps}, std_dev={std_dev}, "
                              f"kernel_size={kernel_size}, lr={lr}, epochs={num_epochs}")
                        
                        # Initialize model
                        model = MyExtremeLearningMachine(
                            num_feature_maps=num_feature_maps,
                            num_classes=10,
                            std_dev=std_dev,
                            feature_size=feature_size,
                            kernel_size=kernel_size
                        )
                        model = model.to(device)
                        
                        # Train model
                        stats = fit_elm_sgd(
                            model=model, 
                            train_loader=train_loader,
                            lr=lr,
                            device=device,
                            num_epochs=num_epochs
                        )
                        
                        # Evaluate model
                        test_accuracy = evaluate(model, test_loader, device)
                        print(f"Test Accuracy: {test_accuracy:.2f}%")
                        
                        # Track results
                        results.append({
                            'num_feature_maps': num_feature_maps,
                            'std_dev': std_dev,
                            'kernel_size': kernel_size,
                            'lr': lr,
                            'num_epochs': num_epochs,
                            'final_train_acc': stats['train_acc'][-1] * 100,
                            'test_acc': test_accuracy
                        })
    
    # Sort results by test accuracy
    results.sort(key=lambda x: x['test_acc'], reverse=True)
    
    # Display top 5 results
    print("\nTop 5 Configurations:")
    for i, result in enumerate(results[:5]):
        print(f"Rank {i+1}:")
        print(f"  Feature Maps: {result['num_feature_maps']}")
        print(f"  Std Dev: {result['std_dev']}")
        print(f"  Kernel Size: {result['kernel_size']}")
        print(f"  Learning Rate: {result['lr']}")
        print(f"  Epochs: {result['num_epochs']}")
        print(f"  Train Accuracy: {result['final_train_acc']:.2f}%")
        print(f"  Test Accuracy: {result['test_acc']:.2f}%")
        print()
    
    # Plot top result training curve
    best_config = results[0]
    print(f"Best Configuration: feature_maps={best_config['num_feature_maps']}, "
          f"std_dev={best_config['std_dev']}, kernel_size={best_config['kernel_size']}, "
          f"lr={best_config['lr']}, epochs={best_config['num_epochs']}")
    
    return results

# Focused experiment with a smaller grid
def focused_experiment():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader = load_cifar10(batch_size=128)
    
    # Promising configurations for ELM without regularization
    configs = [
        # Higher capacity models with moderate std_dev
        {'num_feature_maps': 256, 'std_dev': 0.1, 'kernel_size': 3, 'lr': 0.01, 'num_epochs': 20},
        {'num_feature_maps': 128, 'std_dev': 0.1, 'kernel_size': 3, 'lr': 0.01, 'num_epochs': 25},
        # Larger kernel with lower std_dev
        {'num_feature_maps': 128, 'std_dev': 0.05, 'kernel_size': 5, 'lr': 0.01, 'num_epochs': 20},
        # Higher learning rate for faster convergence
        {'num_feature_maps': 128, 'std_dev': 0.1, 'kernel_size': 3, 'lr': 0.05, 'num_epochs': 15},
        # Lower learning rate with more epochs
        {'num_feature_maps': 128, 'std_dev': 0.1, 'kernel_size': 3, 'lr': 0.005, 'num_epochs': 30}
    ]
    
    results = []
    
    for config in configs:
        print(f"\nTesting configuration: {config}")
        
        # Calculate feature size
        h_out = 32 - config['kernel_size'] + 1
        w_out = 32 - config['kernel_size'] + 1
        feature_size = config['num_feature_maps'] * h_out * w_out
        
        # Initialize model
        model = MyExtremeLearningMachine(
            num_feature_maps=config['num_feature_maps'],
            num_classes=10,
            std_dev=config['std_dev'],
            feature_size=feature_size,
            kernel_size=config['kernel_size']
        )
        model = model.to(device)
        
        # Train model
        stats = fit_elm_sgd(
            model=model, 
            train_loader=train_loader,
            lr=config['lr'],
            device=device,
            num_epochs=config['num_epochs']
        )
        
        # Evaluate model
        test_accuracy = evaluate(model, test_loader, device)
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        
        # Track results
        config_result = config.copy()
        config_result['train_acc'] = stats['train_acc']
        config_result['train_loss'] = stats['train_loss']
        config_result['final_train_acc'] = stats['train_acc'][-1] * 100
        config_result['test_acc'] = test_accuracy
        results.append(config_result)
        
        # Plot training curve using PIL
        img_width, img_height = 800, 400
        plot_image = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(plot_image)
        
        # Draw border and grid lines
        draw.rectangle([50, 50, img_width-50, img_height-50], outline='black')
        
        # Calculate scaling factors
        epochs = config['num_epochs']
        x_scale = (img_width - 100) / epochs
        
        # Plot loss values
        loss_values = stats['train_loss']
        max_loss = max(loss_values)
        loss_y_scale = (img_height - 100) / max_loss
        
        # Draw loss curve
        for i in range(1, epochs):
            x1 = 50 + (i-1) * x_scale
            y1 = img_height - 50 - loss_values[i-1] * loss_y_scale
            x2 = 50 + i * x_scale
            y2 = img_height - 50 - loss_values[i] * loss_y_scale
            draw.line([x1, y1, x2, y2], fill='blue', width=2)
        
        # Add labels
        draw.text((img_width//2, 30), f"Training Curve (FM:{config['num_feature_maps']}, SD:{config['std_dev']}, K:{config['kernel_size']})", fill='black')
        draw.text((img_width//2, img_height-20), "Epochs", fill='black')
        draw.text((20, img_height//2), "Loss / Accuracy", fill='black')
        
        # Save the image
        plot_image.save(f"elm_training_curve_{config['num_feature_maps']}_{config['std_dev']}_{config['kernel_size']}.png")
    
    # Sort results by test accuracy
    results.sort(key=lambda x: x['test_acc'], reverse=True)
    
    # Display results
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"Config {i+1}:")
        print(f"  Feature Maps: {result['num_feature_maps']}")
        print(f"  Std Dev: {result['std_dev']}")
        print(f"  Kernel Size: {result['kernel_size']}")
        print(f"  Learning Rate: {result['lr']}")
        print(f"  Epochs: {result['num_epochs']}")
        print(f"  Final Train Accuracy: {result['final_train_acc']:.2f}%")
        print(f"  Test Accuracy: {result['test_acc']:.2f}%")
        print()
    
    return results

if __name__ == "__main__":
    # Run focused experiment (faster)
    # focused_results = focused_experiment()
    
    # Alternatively, run full grid search (much slower)
    full_results = experiment_hyperparameters()