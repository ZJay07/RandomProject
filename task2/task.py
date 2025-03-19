import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from task2.ensemble_elm import MyEnsembleELM
from task2.metrics_and_visualisation import (
    compute_f1_score,
    evaluate_metrics,
    summarize_metrics,
)
from task2.mix_up import MyMixUp
from task2.montage import visualize_model_predictions
from task2.my_elm import MyExtremeLearningMachine, fit_elm_sgd

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_cifar10(batch_size=128):
    """
    Load and prepare CIFAR-10 dataset for training and testing.
    
    Args:
        batch_size (int, optional): Batch size for DataLoaders. Defaults to 128.
        
    Returns:
        tuple: A tuple containing:
            - train_loader (torch.utils.data.DataLoader): DataLoader for training data
            - test_loader (torch.utils.data.DataLoader): DataLoader for test data
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def evaluate(model, test_loader, device):
    """
    Evaluate a model's accuracy on test data.
    
    Args:
        model (nn.Module): The model to evaluate
        test_loader (torch.utils.data.DataLoader): DataLoader for test data
        device (str or torch.device): Device to run evaluation on
        
    Returns:
        float: Accuracy percentage on test data
    """
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


def experiment_hyperparameters(save_path="./task2/models/hyperprameters_for_sgd"):
    """
    Perform hyperparameter search for Extreme Learning Machine models.
    
    Systematically explores combinations of feature maps, standard deviation,
    kernel size, learning rate, and training epochs to find optimal settings.
    
    Args:
        save_path (str, optional): Directory to save best model and results.
                                  Defaults to "./task2/models/hyperprameters_for_sgd".
        
    Returns:
        list: Results of all hyperparameter combinations tested, sorted by test accuracy
    """
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, test_loader = load_cifar10(batch_size=128)

    # Hyperparameter grid
    feature_maps_options = [64, 128]
    std_dev_options = [0.01, 0.1, 0.5, 1.0]
    kernel_size_options = [3, 5, 7]
    lr_options = [0.01, 0.1]
    epochs_options = [10, 20, 50]

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
                        print(
                            f"\nTesting: feature_maps={num_feature_maps}, std_dev={std_dev}, "
                            f"kernel_size={kernel_size}, lr={lr}, epochs={num_epochs}"
                        )

                        # Initialize model
                        model = MyExtremeLearningMachine(
                            num_feature_maps=num_feature_maps,
                            num_classes=10,
                            std_dev=std_dev,
                            feature_size=feature_size,
                            kernel_size=kernel_size,
                        )
                        model = model.to(device)

                        # Train model
                        stats, final_metrics = fit_elm_sgd(
                            model=model,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            lr=lr,
                            device=device,
                            num_epochs=num_epochs,
                        )

                        # Evaluate model
                        test_accuracy = evaluate(model, test_loader, device)
                        print(f"Test Accuracy: {test_accuracy:.2f}%")

                        # Track results
                        results.append(
                            {
                                "num_feature_maps": num_feature_maps,
                                "std_dev": std_dev,
                                "kernel_size": kernel_size,
                                "lr": lr,
                                "num_epochs": num_epochs,
                                "final_train_acc": stats["train_acc"][-1] * 100,
                                "test_acc": test_accuracy,
                            }
                        )
    # Sort results by test accuracy
    results.sort(key=lambda x: x["test_acc"], reverse=True)

    # Display top 5 results
    print("\nTop 5 Configurations:")
    for i, result in enumerate(results[:5]):
        print(f"Rank {i + 1}:")
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
    print(
        f"Best Configuration: feature_maps={best_config['num_feature_maps']}, "
        f"std_dev={best_config['std_dev']}, kernel_size={best_config['kernel_size']}, "
        f"lr={best_config['lr']}, epochs={best_config['num_epochs']}"
    )

    # saving the configs
    os.makedirs("./models", exist_ok=True)
    h_out = 32 - best_config["kernel_size"] + 1
    w_out = 32 - best_config["kernel_size"] + 1
    feature_size = best_config["num_feature_maps"] * h_out * w_out

    best_model = MyExtremeLearningMachine(
        num_feature_maps=best_config["num_feature_maps"],
        num_classes=10,
        std_dev=best_config["std_dev"],
        feature_size=feature_size,
        kernel_size=best_config["kernel_size"],
    )
    best_model = best_model.to(device)

    # Train model with best configuration
    best_stats, best_metrics = fit_elm_sgd(
        model=best_model,
        train_loader=train_loader,
        test_loader=test_loader,
        lr=best_config["lr"],
        device=device,
        num_epochs=best_config["num_epochs"],
    )

    # Save best model
    os.mkdir(save_path)
    best_model_path = os.path.join(save_path, "best_hyperparameter_model.pth")
    torch.save(best_model.state_dict(), best_model_path)
    print(f"Best model saved to {best_model_path}")

    # Save configs details for loading
    best_config_path = os.path.join(save_path, "best_hyperparameter_config.json")
    with open(best_config_path, "w") as f:
        json.dump(
            {
                "num_feature_maps": best_config["num_feature_maps"],
                "std_dev": best_config["std_dev"],
                "kernel_size": best_config["kernel_size"],
                "feature_size": feature_size,
            },
            f,
            indent=4,
        )
    print(f"Best configuration saved to {best_config_path}")
    results_json_path = os.path.join(save_path, "hyperparameter_results.json")
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_json_path}")

    return results


def train_with_mixup(
    model,
    train_loader,
    test_loader,
    mixup,
    lr=0.01,
    device="cpu",
    num_epochs=10,
    eval_every=1,
):
    """
    Train a model using MixUp data augmentation.
    
    Args:
        model (nn.Module): The model to train
        train_loader (torch.utils.data.DataLoader): DataLoader for training data
        test_loader (torch.utils.data.DataLoader): DataLoader for test data
        mixup (MyMixUp): MixUp augmentation instance
        lr (float, optional): Learning rate for SGD. Defaults to 0.01.
        device (str, optional): Device to train on. Defaults to "cpu".
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

            # Apply MixUp
            mixed_inputs, targets_a, targets_b, lam = mixup(inputs, targets, device)

            optimizer.zero_grad()
            outputs = model(mixed_inputs)

            # Calculate mixed loss
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(
                outputs, targets_b
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            # For accuracy calculation (use original inputs)
            with torch.no_grad():
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
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


def train_ensemble(
    ensemble_model,
    train_loader,
    test_loader,
    fit_function,
    lr=0.01,
    device="cpu",
    num_epochs=10,
    eval_every=1,
):
    """
    Train an ensemble of models while tracking metrics.
    
    Trains each model in the ensemble sequentially and evaluates ensemble
    performance after each model is trained.
    
    Args:
        ensemble_model (MyEnsembleELM): The ensemble model to train
        train_loader (torch.utils.data.DataLoader): DataLoader for training data
        test_loader (torch.utils.data.DataLoader): DataLoader for test data
        fit_function (callable): Function used to train each model
        lr (float, optional): Learning rate. Defaults to 0.01.
        device (str, optional): Device to train on. Defaults to "cpu".
        num_epochs (int, optional): Number of training epochs per model. Defaults to 10.
        eval_every (int, optional): Frequency of ensemble evaluation. Defaults to 1.
        
    Returns:
        tuple: A tuple containing:
            - statistics (dict): Dictionary with ensemble metrics over time
            - final_metrics (dict): Dictionary with final evaluation metrics
    """
    ensemble_model = ensemble_model.to(device)

    statistics = {"test_acc": [], "test_f1": [], "epochs": []}

    # Train each model in the ensemble
    for i, model in enumerate(ensemble_model.models):
        print(f"Training ensemble model {i + 1}/{ensemble_model.n_models}")
        fit_function(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            lr=lr,
            device=device,
            num_epochs=num_epochs,
        )

        # Evaluate ensemble after each model is trained
        if (i + 1) % eval_every == 0 or i == ensemble_model.n_models - 1:
            # For ensemble evaluation, we need to use the ensemble's eval method
            ensemble_acc = ensemble_model.eval(test_loader, device) * 100

            # Calculate F1 score
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = ensemble_model(inputs)
                    _, predicted = torch.max(outputs, 1)

                    all_preds.extend(predicted.cpu().tolist())
                    all_targets.extend(targets.cpu().tolist())

            ensemble_f1 = 100 * compute_f1_score(
                all_targets, all_preds, average="macro"
            )

            statistics["test_acc"].append(ensemble_acc)
            statistics["test_f1"].append(ensemble_f1)
            statistics["epochs"].append((i + 1) * num_epochs)

            print(
                f"Ensemble Model ({i + 1}/{ensemble_model.n_models}), Test Acc: {ensemble_acc:.2f}%, Test F1: {ensemble_f1:.2f}%"
            )

    # final eval
    final_ensemble_acc = ensemble_model.eval(test_loader, device) * 100

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = ensemble_model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    final_ensemble_f1 = 100 * compute_f1_score(all_targets, all_preds, average="macro")

    final_metrics = {
        "accuracy": final_ensemble_acc,
        "macro_f1": final_ensemble_f1,
    }

    return statistics, final_metrics


def experiment_regularization_methods(
    num_feature_maps=128,
    std_dev=0.01,
    kernel_size=7,
    lr=0.1,
    num_epochs=50,
    num_ensemble_models=5,
    save_dir="./task2/models",
    metric_dir="./task2/metrics",
):
    """
    Experiment with different regularization techniques for ELM models.
    
    Compares performance of four approaches:
    1. Base ELM (no regularization)
    2. MixUp data augmentation
    3. Ensemble ELM
    4. Ensemble ELM with MixUp
    
    Args:
        num_feature_maps (int, optional): Number of feature maps. Defaults to 128.
        std_dev (float, optional): Standard deviation for weight initialization. Defaults to 0.01.
        kernel_size (int, optional): Size of convolutional kernel. Defaults to 7.
        lr (float, optional): Learning rate. Defaults to 0.1.
        num_epochs (int, optional): Number of training epochs. Defaults to 50.
        num_ensemble_models (int, optional): Number of models in ensemble. Defaults to 5.
        save_dir (str, optional): Directory to save models. Defaults to "./task2/models".
        metric_dir (str, optional): Directory to save metrics. Defaults to "./task2/metrics".
        
    Returns:
        dict: Results dictionary with performance metrics for all methods
    """
    # Create directory for saving models if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader, test_loader = load_cifar10(batch_size=128)

    # Calculate feature size for CIFAR-10
    h_out = 32 - kernel_size + 1
    w_out = 32 - kernel_size + 1
    feature_size = num_feature_maps * h_out * w_out

    # Initialize MixUp
    mixup = MyMixUp(alpha=1.0, seed=SEED)

    results = {}

    # 1. Train a base ELM for comparison
    print("\n=== Training Base ELM Model ===")
    base_model = MyExtremeLearningMachine(
        num_feature_maps=num_feature_maps,
        num_classes=10,
        std_dev=std_dev,
        feature_size=feature_size,
        kernel_size=kernel_size,
    )
    base_model = base_model.to(device)

    base_stats, base_final_metrics = fit_elm_sgd(
        model=base_model,
        train_loader=train_loader,
        test_loader=test_loader,
        lr=lr,
        device=device,
        num_epochs=num_epochs,
    )

    base_accuracy = evaluate(base_model, test_loader, device)
    print(f"Base Model Test Accuracy: {base_accuracy:.2f}%")

    # Save base model
    base_model_path = os.path.join(save_dir, "base_model.pth")
    torch.save(base_model.state_dict(), base_model_path)
    print(f"Base model saved to {base_model_path}")
    results["base_model"] = {
        "path": base_model_path,
        "epochs": base_stats["epochs"],
        "test_acc": base_stats["test_acc"],
        "test_f1": base_stats["test_f1"],
        "final_metrics": base_final_metrics,
    }

    # 2. Train a model using only MixUp
    print("\n=== Training Model with MixUp ===")
    mixup_model = MyExtremeLearningMachine(
        num_feature_maps=num_feature_maps,
        num_classes=10,
        std_dev=std_dev,
        feature_size=feature_size,
        kernel_size=kernel_size,
    )
    mixup_model = mixup_model.to(device)

    mixup_stats, mixup_final_metrics = train_with_mixup(
        model=mixup_model,
        train_loader=train_loader,
        test_loader=test_loader,
        mixup=mixup,
        lr=lr,
        device=device,
        num_epochs=num_epochs,
    )

    mixup_accuracy = evaluate(mixup_model, test_loader, device)
    print(f"MixUp Model Test Accuracy: {mixup_accuracy:.2f}%")

    # Save mixup model
    mixup_model_path = os.path.join(save_dir, "mixup_model.pth")
    torch.save(mixup_model.state_dict(), mixup_model_path)
    print(f"MixUp model saved to {mixup_model_path}")
    results["mixup_model"] = {
        "path": mixup_model_path,
        "epochs": mixup_stats["epochs"],
        "test_acc": mixup_stats["test_acc"],
        "test_f1": mixup_stats["test_f1"],
        "final_metrics": mixup_final_metrics,
    }

    # 3. Train a model using only Ensemble ELM
    print("\n=== Training Ensemble ELM Model ===")
    ensemble_model = MyEnsembleELM(
        seed=SEED,
        n_models=num_ensemble_models,
        num_feature_maps=num_feature_maps,
        kernel_size=kernel_size,
        std_dev=std_dev,
    )
    ensemble_model = ensemble_model.to(device)

    # Train the ensemble
    ensemble_stats, ensemble_final_metrics = train_ensemble(
        ensemble_model=ensemble_model,
        train_loader=train_loader,
        test_loader=test_loader,
        fit_function=fit_elm_sgd,
        lr=lr,
        device=device,
        num_epochs=num_epochs,
        eval_every=1,
    )

    ensemble_accuracy = ensemble_model.eval(test_loader, device) * 100
    print(f"Ensemble ELM Test Accuracy: {ensemble_accuracy:.2f}%")

    # Save ensemble model
    # For ensemble, we need to save each model in the ensemble
    ensemble_dir = os.path.join(save_dir, "ensemble_model")
    os.makedirs(ensemble_dir, exist_ok=True)
    for i, model in enumerate(ensemble_model.models):
        model_path = os.path.join(ensemble_dir, f"model_{i}.pth")
        torch.save(model.state_dict(), model_path)

    # Save ensemble configuration
    ensemble_config_path = os.path.join(save_dir, "ensemble_config.pth")
    torch.save(
        {
            "n_models": ensemble_model.n_models,
            "seed": ensemble_model.seed,
            "num_feature_maps": num_feature_maps,
            "std_dev": std_dev,
            "kernel_size": kernel_size,
        },
        ensemble_config_path,
    )
    print(f"Ensemble model saved to {ensemble_dir}")
    results["ensemble_model"] = {
        "dir": ensemble_dir,
        "config": ensemble_config_path,
        "epochs": ensemble_stats["epochs"],
        "test_acc": ensemble_stats["test_acc"],
        "test_f1": ensemble_stats["test_f1"],
        "final_metrics": ensemble_final_metrics,
    }

    # 4. Train a model using both MixUp and Ensemble ELM
    print("\n=== Training Model with MixUp and Ensemble ELM ===")
    ensemble_mixup_model = MyEnsembleELM(
        seed=SEED,
        n_models=num_ensemble_models,
        num_feature_maps=num_feature_maps,
        kernel_size=kernel_size,
        std_dev=std_dev,
    )
    ensemble_mixup_model = ensemble_mixup_model.to(device)

    ensemble_mixup_stats = {"test_acc": [], "test_f1": [], "epochs": []}

    # Train each model in the ensemble with MixUp
    for i, model in enumerate(ensemble_mixup_model.models):
        print(
            f"Training ensemble model {i + 1}/{ensemble_mixup_model.n_models} with MixUp"
        )
        _, _ = train_with_mixup(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            mixup=mixup,
            lr=lr,
            device=device,
            num_epochs=num_epochs,
            eval_every=num_epochs,
        )

        # Evaluate the ensemble after each model is trained
        ensemble_mixup_accuracy = ensemble_mixup_model.eval(test_loader, device) * 100

        # Calculate F1 score for the ensemble
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = ensemble_mixup_model(inputs)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().tolist())
                all_targets.extend(targets.cpu().tolist())

        ensemble_mixup_f1 = 100 * compute_f1_score(
            all_targets, all_preds, average="macro"
        )

        ensemble_mixup_stats["test_acc"].append(ensemble_mixup_accuracy)
        ensemble_mixup_stats["test_f1"].append(ensemble_mixup_f1)
        ensemble_mixup_stats["epochs"].append((i + 1) * num_epochs)

        print(
            f"Ensemble with MixUp ({i + 1}/{ensemble_mixup_model.n_models}), Test Acc: {ensemble_mixup_accuracy:.2f}%, Test F1: {ensemble_mixup_f1:.2f}%"
        )

    # Final evaluation for ensemble mixup
    ensemble_mixup_acc = ensemble_mixup_model.eval(test_loader, device) * 100

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = ensemble_mixup_model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    ensemble_mixup_f1 = 100 * compute_f1_score(all_targets, all_preds, average="macro")

    ensemble_mixup_final_metrics = {
        "accuracy": ensemble_mixup_acc,
        "macro_f1": ensemble_mixup_f1,
    }

    # Save ensemble mixup model
    ensemble_mixup_dir = os.path.join(save_dir, "ensemble_mixup_model")
    os.makedirs(ensemble_mixup_dir, exist_ok=True)
    for i, model in enumerate(ensemble_mixup_model.models):
        model_path = os.path.join(ensemble_mixup_dir, f"model_{i}.pth")
        torch.save(model.state_dict(), model_path)

    # Save ensemble mixup configuration
    ensemble_mixup_config_path = os.path.join(save_dir, "ensemble_mixup_config.pth")
    torch.save(
        {
            "n_models": ensemble_mixup_model.n_models,
            "seed": ensemble_mixup_model.seed,
            "num_feature_maps": num_feature_maps,
            "std_dev": std_dev,
            "kernel_size": kernel_size,
            "mixup_alpha": mixup.alpha,
        },
        ensemble_mixup_config_path,
    )
    print(f"Ensemble MixUp model saved to {ensemble_mixup_dir}")

    results["ensemble_mixup_model"] = {
        "dir": ensemble_mixup_dir,
        "config": ensemble_mixup_config_path,
        "epochs": ensemble_mixup_stats["epochs"],
        "test_acc": ensemble_mixup_stats["test_acc"],
        "test_f1": ensemble_mixup_stats["test_f1"],
        "final_metrics": ensemble_mixup_final_metrics,
    }

    # Compare all results
    print("\n=== Experiment Results ===")
    print(f"Base Model Accuracy: {base_accuracy:.2f}%")
    print(f"MixUp Model Accuracy: {mixup_accuracy:.2f}%")
    print(f"Ensemble ELM Accuracy: {ensemble_accuracy:.2f}%")
    print(f"Ensemble ELM with MixUp Accuracy: {ensemble_mixup_accuracy:.2f}%")

    # Save results summary as JSON
    results_json = {}
    for model_name, results in results.items():
        model_results = {
            "epochs": results["epochs"],
            "test_acc": results["test_acc"],
            "test_f1": results["test_f1"],
            "final_accuracy": results["final_metrics"]["accuracy"],
            "final_f1": results["final_metrics"]["macro_f1"],
        }
        results_json[model_name] = model_results

    with open(
        os.path.join(metric_dir, "metrics_results_regularasation_method.json"), "w"
    ) as f:
        json.dump(results_json, f, indent=4)

    # Viz for summary of report
    json_file_path = os.path.join(
        metric_dir, "metrics_results_regularasation_method.json"
    )

    if os.path.exists(json_file_path):
        print(f"Found metrics file: {json_file_path}")
        # generate summary with helper function
        summarize_metrics(json_file_path)
    else:
        print(f"Metrics file not found: {json_file_path}")
        print(
            "Please make sure you've run the experiments and generated the JSON file."
        )

    return results


if __name__ == "__main__":
    print("=== What is considered 'random guess' in multiclass classification? ===")
    print("A random guess in multiclass classification involves assigning labels by pure chance, with equal probability across all classes. In balanced datasets with K classes, the expected accuracy is 1/K (e.g., 10% for 10 classes). For imbalanced datasets, the baseline becomes the frequency of the most common class. To test if a model performs better than random guessing, statistical tests (chi-squared, binomial) determine if the observed accuracy significantly exceeds this baseline. This represents the minimum performance threshold a useful classifier must surpass.")
    
    print("=== Hyperparameter Experimentation Without Regularisation ===")
    print("*Due to the nature of the search space, this may take awhile*")
    experiment_hyperparameters()

    print("=== Regularisation Methods Experimentation ===")
    print("*pre-ran summary in 'task2/logs/summary_of_regularisation_methods.txt'*")
    experiment_regularization_methods()
    print("=== Metric justifications: ===")
    print("Accuracy provies an overall performance measure, especially useful when classes are balanced like in CIFAR-10")
    print("Macro-F1 score balances precision and recall across all classes, detecting if certain classes are more challenging to classify regardless of their frequency.")
    
    
    # Paths to regularisation model file
    best_regularisation_model_dir = "./task2/models/ensemble_model"
    best_regularisation_config_file = "./task2/models/ensemble_config.pth"

    # viz for best regularisation method
    save_path = "./task2/montage_result/result.png"
    visualize_model_predictions(best_regularisation_model_dir, best_regularisation_config_file, save_path=save_path)
    print(f"Visualisation saved to {save_path}")
