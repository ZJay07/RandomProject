import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from  task2.metrics import evaluate_metrics
from task2.task import load_cifar10
from task2.ensemble_elm import MyEnsembleELM
from task2.my_elm import fit_elm_sgd
import numpy as np
import json


SEED = 42
ENSEMBLE_SIZE = 10

# TODO: Add new metrics, and consider adding loss as well for viz 
def fit_elm_ls(model, train_loader, test_loader=None, lambda_reg=0.1, device="cpu", method="ridge"):
    """
    Using a variant of the least squares algorithm to train the ELM - ridge
    device should always be CPU due to the library constrains but just in case cuda is used
    batch implementation to save memory
    """
    # Record training start time
    model = model.to(device)
    model.eval()  # ensure fixed layers are in eval mode
    
    H_list = []
    T_list = []
    statistics = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "test_f1": [],
        "epochs": [1]
    }
    
    # Accumulate the hidden features and one-hot encoded targets batch-by-batch
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            
            # Forward pass through the fixed conv layers and ReLU
            x = model.conv(inputs)
            x = F.relu(x)
            x = x.view(x.size(0), -1)  # flatten to shape: [batch_size, feature_dim]
            H_list.append(x)
            
            # Create one-hot encoded targets for this batch
            batch_size = x.size(0)
            T_batch = torch.zeros(batch_size, model.num_classes, device=device)
            T_batch.scatter_(1, targets.to(device).unsqueeze(1), 1)
            T_list.append(T_batch)
    
    # Concatenate all batches: H: (N, d), T: (N, num_classes)
    H = torch.cat(H_list, dim=0)
    T = torch.cat(T_list, dim=0)
    
    N = H.shape[0]  # number of training samples
    
    # Compute the inverse of the smaller matrix: (H H^T + lambda_reg * I_N)
    I_N = torch.eye(N, device=device)
    inv_term = torch.linalg.inv(H @ H.T + lambda_reg * I_N)
    
    # Compute beta using the Woodbury identity:
    # beta = H^T * (H H^T + lambda_reg I_N)^(-1) * T
    beta = H.T @ (inv_term @ T)
    
    print(f"Computed beta shape (weights): {beta.shape}")
    
    # Step 3: Set the computed weights to the output layer
    with torch.no_grad():
        model.fc.weight.copy_(beta.t())  # Transpose because PyTorch uses weight * input
        if model.fc.bias is not None:
            model.fc.bias.zero_()  # Zero out the bias
    
    # Step 4: Evaluate on training data
    model.eval()
    correct = 0
    total = 0
    train_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    print("Evaluating on training data...")
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    train_loss = train_loss / total
    train_acc = correct / total
    statistics["train_loss"].append(train_loss)
    statistics["train_acc"].append(train_acc)
    
    # Step 5: Evaluate on test data
    print("Evaluating on test data...")
    metrics = evaluate_metrics(model, test_loader, device)
    statistics["test_acc"].append(metrics['accuracy'])
    statistics["test_f1"].append(metrics['macro_f1'])
    
    print(f"Direct solution ({method}) - Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Test Acc: {metrics['accuracy']:.2f}%, Test F1: {metrics['macro_f1']:.2f}%")
    
    return statistics, metrics
    

# TODO: Complete this, need to use best model configurations (ensemble ELM and best hyperparamters )
def comparison_duration_sgd_and_ls(feature_maps, std_dev, kernel_size,  lr, epoch, ensemble_size = 10, ls_lambda = 0.001, device = 'cpu'):
    """Comparision with best model - ensemble ELM"""

    print("Comparing ls and sgd training duration...")
    print(f"Using device: {device}")
    

    # globals
    h_out = 32 - kernel_size + 1
    w_out = 32 - kernel_size + 1
    feature_size = feature_maps * h_out * w_out 

    train_loader, test_loader = load_cifar10(batch_size=128)

    # set seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("\n=== Training with Least Squares ===")
    ls_model = MyEnsembleELM(
        seed=SEED,
        n_models=ensemble_size,
        num_feature_maps=feature_maps,
        std_dev=std_dev
    )
    ls_model.to(device)

    ls_start = time.time()
    # train each model in the ensemble individually
    for i, model in enumerate(ls_model.models):
        print(f"Training model {i+1}/{ensemble_size} with LS")
        _, _ = fit_elm_ls(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            lambda_reg=ls_lambda, 
            device=device
        )
    ls_metrics = evaluate_metrics(ls_model, test_loader, device)
    ls_time = time.time() - ls_start

    print("\n=== Training with SGD ===")
    sgd_model = MyEnsembleELM(
        seed=SEED,
        n_models=ensemble_size,
        num_feature_maps=feature_maps,
        std_dev=std_dev
    )

    sgd_start = time.time()
    # train each model in the ensemble individually with SGD
    for i, model in enumerate(sgd_model.models):
        print(f"Training model {i+1}/{ensemble_size} with SGD")
        _, _ = fit_elm_sgd(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            lr=lr,
            device=device,
            num_epochs=epoch
        )
    sgd_metrics = evaluate_metrics(sgd_model, test_loader, device)
    sgd_time = time.time() - sgd_start

    print("\n=== Method Comparison ===")
    print("Training time comparison between Least Squares and SGD:")
    print(f"Least Squares Training Time: {ls_time:.2f}s")
    print(f"SGD Training Time: {sgd_time:.2f}s")
    print()
    print("Metrics comparison:")
    print(f"LS Test Acc: {ls_metrics['accuracy']:.2f}%, Test F1: {ls_metrics['macro_f1']:.2f}%")
    print(f"SGD Test Acc: {sgd_metrics['accuracy']:.2f}%, Test F1: {sgd_metrics['macro_f1']:.2f}%")

    return {
        "ls": {"accuracy": ls_metrics['accuracy'], "Test F1": ls_metrics['macro_f1'], "time": ls_time},
        "sgd": {"accuracy": sgd_metrics['accuracy'], "Test F1": sgd_metrics['macro_f1'], "time": sgd_time}
    }


def random_search_hyperparameter_ls(
        num_steps = 100,
        train_loader=None,
        test_loader=None,
        feature_maps_range=(64, 128),
        std_dev_range=(0.01, 0.5),
        kernel_sizes=[3, 5, 7],
        lambda_range=(1e-5, 0.1),
        ensemble_size=(3, 15),
        seed = 42,
        save_dir="./models"
):
    """
    Using random search and ls to find best hyperparameters, similar to task2 hyperparamters tuned: std, feature maps, kernel size, lambda (for ls)
    Using defined search space from hyperparameter arguments
    """
    # default should be cpu due to environment constraints
    device = 'cpu'

    # set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    results = []
    best_acc = 0
    best_config = None
    best_ensemble = None

    for step in range(num_steps):
        # Randomly sample hyperparameters
        n_models = int(np.random.uniform(ensemble_size[0], ensemble_size[1]))
        num_feature_maps = int(np.exp(np.random.uniform(np.log(feature_maps_range[0]), 
                                                      np.log(feature_maps_range[1]))))
        std_dev = np.random.uniform(std_dev_range[0], std_dev_range[1])
        kernel_size = np.random.choice(kernel_sizes)
        lambda_reg = np.exp(np.random.uniform(np.log(lambda_range[0]), 
                                             np.log(lambda_range[1])))

        print(f"Step {step+1}/{num_steps}:")
        print(f"Feature maps: {num_feature_maps}, Std Dev: {std_dev}, Kernel Size: {kernel_size}, Lambda: {lambda_reg}")

        # Train model using least squares
        try:
            step_seed = seed + step # unique seed for each step, fixed for reproducibility
            ensemble = MyEnsembleELM(
                seed=step_seed,
                n_models=n_models,
                num_feature_maps=num_feature_maps,
                std_dev=std_dev,
            )
            ensemble.to(device)
            start_time = time.time()
            acc_progression = []
            f1_progression = []
            for i, model in enumerate(ensemble.models):
                print(f"Training model {i+1}/{n_models}")
                
                # Train with least squares
                _, _ = fit_elm_ls(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    lambda_reg=lambda_reg,
                    device=device
                )
                
                # Evaluate current ensemble state
                metrics = evaluate_metrics(ensemble, test_loader, device)
                acc_progression.append(metrics["accuracy"])
                f1_progression.append(metrics["macro_f1"])
                
                print(f"  Ensemble accuracy after {i+1} models: {metrics['accuracy']:.2f}%")
            
            training_time = time.time() - start_time
            final_accuracy = acc_progression[-1]
            final_f1 = f1_progression[-1]
            result = {
                "n_models": ensemble_size,
                "num_feature_maps": num_feature_maps,
                "std_dev": std_dev,
                "lambda_reg": lambda_reg,
                "accuracy": final_accuracy,
                "f1": final_f1,
                "training_time": training_time,
                "acc_progression": acc_progression,
                "f1_progression": f1_progression
            }
            results.append(result)
            print(f"Results: Accuracy={final_accuracy:.2f}%, F1={final_f1:.2f}%, Time={training_time:.2f}s")

            if final_accuracy > best_acc:
                best_acc = final_accuracy
                best_config = result.copy()
                best_ensemble = ensemble
                print(f"New best model found! Accuracy: {best_acc:.2f}%")
        
        except Exception as e:
            print(f"Training failed: {e}")
            continue

    # sort best results
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)

    print("\n=== Top 5 Configurations ===")
    for i, result in enumerate(sorted_results[:5]):
        print(f"Rank {i+1}:")
        print(f"  Ensemble Size: {result['n_models']}")
        print(f"  Feature Maps: {result['num_feature_maps']}")
        print(f"  Std Dev: {result['std_dev']:.6f}")
        print(f"  Lambda: {result['lambda_reg']:.6f}")
        print(f"  Accuracy: {result['accuracy']:.2f}%")
        print(f"  F1 Score: {result['f1']:.2f}%")
        print(f"  Training Time: {result['training_time']:.2f}s")
        print()

    # Save results to json
    results_json_path = os.path.join(save_dir, 'elm_ls_hyperparameter_search_results.json')
    with open(results_json_path, 'w') as f:
        json.dump(sorted_results, f, indent=4)
    print(f"Results saved to '{results_json_path}'")
    
    # Save best model
    best_model_path = os.path.join(save_dir, "best_hyperparameter_model.pth")
    torch.save(best_ensemble.state_dict(), best_model_path)
    print(f"Best model saved to {best_model_path}")
    
    # Save config details for loading
    best_config_path = os.path.join(save_dir, "best_hyperparameter_config.json")
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=4)
    print(f"Best configuration saved to {best_config_path}")
    
    # Compare with previous best
    previous_best = 54.89  # previous best ensemble accuracy
    print(f"\nPrevious best Ensemble ELM: {previous_best:.2f}%")
    print(f"New best Ensemble ELM: {best_acc:.2f}%")
    print(f"Improvement: {best_acc - previous_best:.2f}%")

    return sorted_results, best_ensemble, best_config

if __name__ == "__main__":
    print("Comparing training duration for SGD and LS...")
    print("Trying with best parameters from previous experiments")
    # (feature_maps, std_dev, kernel_size,  lr, epoch, ensemble_size = 10, ls_lambda = 0.001)
    comparison_duration_sgd_and_ls(feature_maps = 64, std_dev = 0.1, kernel_size=3, lr = 0.01, epoch=20)

    print("Preparing to run random search for hyperparameters...")
    # train_loader, test_loader = load_cifar10()
    # random_search_hyperparameter_ls(train_loader=train_loader, test_loader=test_loader)