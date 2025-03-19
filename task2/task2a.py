import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json

import numpy as np

from task2.ensemble_elm import MyEnsembleELM
from task2.metrics_and_visualisation import convert_numpy_types, evaluate_metrics
from task2.montage import visualize_model_predictions
from task2.my_elm import fit_elm_sgd
from task2.task import load_cifar10

SEED = 42


def fit_elm_ls(model, train_loader, test_loader=None, lambda_reg=0.1, device="cpu"):
    """
    Using a variant of the least squares algorithm to train the ELM - ridge regression
    device should always be CPU due to the library constrains but just in case cuda is used
    batch implementation to save memory
    """
    # Record training start time
    start_time = time.time()

    print("\nRunning fit elm ls with ridge regression...")
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()

    statistics = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "test_f1": [],
        "epochs": [1],
    }

    # Get feature dimension using the model's forward logic
    with torch.no_grad():
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            x = model.conv(inputs)
            x = F.relu(x)
            if model.pooling:
                x = F.avg_pool2d(x, 2)  # reducing dimensions
            x = x.view(x.size(0), -1)
            feature_dim = x.size(1)
            print(f"Feature dimension: {feature_dim}")
            break

    # Initialize the solution components
    HTH = torch.zeros(feature_dim, feature_dim, device=device)  # H^T H
    HTT = torch.zeros(feature_dim, model.num_classes, device=device)  # H^T T

    # Process data in batches to calculate H^T H and H^T T
    print("Accumulating H^T H and H^T T matrices...")
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            # Forward pass through fixed layers using the model's logic
            x = model.conv(inputs)
            x = F.relu(x)
            if model.pooling:
                x = F.avg_pool2d(x, 2)
            H_batch = x.view(batch_size, -1)

            # Create one-hot targets
            T_batch = torch.zeros(batch_size, model.num_classes, device=device)
            T_batch.scatter_(1, targets.to(device).unsqueeze(1), 1)

            # Update running sums
            HTH += H_batch.T @ H_batch
            HTT += H_batch.T @ T_batch

    # Add regularization
    print(f"Adding regularization (lambda={lambda_reg})")
    HTH += lambda_reg * torch.eye(feature_dim, device=device)

    # Solve the system
    print("Solving linear system...")
    try:
        # Process in batches of classes to save memory
        class_batch_size = 2
        for start_idx in range(0, model.num_classes, class_batch_size):
            end_idx = min(start_idx + class_batch_size, model.num_classes)
            print(f"Solving for classes {start_idx + 1}-{end_idx}/{model.num_classes}")

            HTT_batch = HTT[:, start_idx:end_idx]
            # Direct solver
            beta_batch = torch.linalg.solve(HTH, HTT_batch)

            # Update weights for these classes
            with torch.no_grad():
                model.fc.weight.data[start_idx:end_idx] = beta_batch.T
    except RuntimeError as e:
        print(f"Direct solve failed: {e}")
        print("Fall back: Using SVD approach for better numerical stability")
        try:
            # Potentially fail due to numerical stability, should be rare
            U, S, Vh = torch.linalg.svd(HTH, full_matrices=False)
            S_reg = 1.0 / (S + lambda_reg)

            # Process in batches of classes
            class_batch_size = 2
            for start_idx in range(0, model.num_classes, class_batch_size):
                end_idx = min(start_idx + class_batch_size, model.num_classes)
                print(
                    f"Solving for classes {start_idx + 1}-{end_idx}/{model.num_classes} (SVD)"
                )

                HTT_batch = HTT[:, start_idx:end_idx]
                beta_batch = Vh.T @ (S_reg.unsqueeze(1) * (U.T @ HTT_batch))

                with torch.no_grad():
                    model.fc.weight.data[start_idx:end_idx] = beta_batch.T
        except Exception as e:
            print(f"SVD approach failed: {e}")
            return None, None

    # Free up memory
    HTH = None
    HTT = None

    # Evaluate on training data
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

    # Evaluate on test data
    print("Evaluating on test data...")
    if test_loader:
        metrics = evaluate_metrics(model, test_loader, device)
        statistics["test_acc"].append(metrics["accuracy"])
        statistics["test_f1"].append(metrics["macro_f1"])

        print(
            f"LS solution - Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Test Acc: {metrics['accuracy']:.2f}%, Test F1: {metrics['macro_f1']:.2f}%"
        )
    else:
        metrics = {"accuracy": 0, "macro_f1": 0}
        print(f"LS solution - Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    print(f"This fit took {time.time() - start_time} to complete!")
    return statistics, metrics


def comparison_duration_sgd_and_ls(
    feature_maps,
    std_dev,
    kernel_size,
    lr,
    epoch,
    ensemble_size=5,
    ls_lambda=0.001,
    device="cpu",
    save_dir="./task2/metrics",
):
    """Comparision with best model - ensemble ELM"""

    print("Comparing ls and sgd training duration...")
    print(f"Using device: {device}")

    # Calculate feature sizes for both approaches
    # For SGD (no pooling)
    h_out_sgd = 32 - kernel_size + 1
    w_out_sgd = 32 - kernel_size + 1
    feature_size_sgd = feature_maps * h_out_sgd * w_out_sgd

    # For LS (with pooling)
    h_out_ls = (32 - kernel_size + 1) // 2
    w_out_ls = (32 - kernel_size + 1) // 2
    feature_size_ls = feature_maps * h_out_ls * w_out_ls

    train_loader, test_loader = load_cifar10(batch_size=128)

    # set seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("\n=== Training with Least Squares ===")
    ls_model = MyEnsembleELM(
        seed=SEED,
        n_models=ensemble_size,
        num_feature_maps=feature_maps,
        std_dev=std_dev,
        feature_size=feature_size_ls,
        kernel_size=kernel_size,
        pooling=True,
    )
    ls_model.to(device)

    ls_start = time.time()
    # train each model in the ensemble individually
    for i, model in enumerate(ls_model.models):
        print(f"Training model {i + 1}/{ensemble_size} with LS")
        _, _ = fit_elm_ls(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            lambda_reg=ls_lambda,
            device=device,
        )
    ls_metrics = evaluate_metrics(ls_model, test_loader, device)
    ls_time = time.time() - ls_start

    print("\n=== Training with SGD ===")
    sgd_model = MyEnsembleELM(
        seed=SEED,
        n_models=ensemble_size,
        num_feature_maps=feature_maps,
        std_dev=std_dev,
        feature_size=feature_size_sgd,
        kernel_size=kernel_size,
        pooling=False,
    )

    sgd_start = time.time()
    # train each model in the ensemble individually with SGD
    for i, model in enumerate(sgd_model.models):
        print(f"Training model {i + 1}/{ensemble_size} with SGD")
        _, _ = fit_elm_sgd(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            lr=lr,
            device=device,
            num_epochs=epoch,
        )
    sgd_metrics = evaluate_metrics(sgd_model, test_loader, device)
    sgd_time = time.time() - sgd_start

    print("\n=== Method Comparison ===")
    print("Training time comparison between Least Squares and SGD:")
    print(f"Least Squares Training Time: {ls_time:.2f}s")
    print(f"SGD Training Time: {sgd_time:.2f}s")
    print()
    print("Metrics comparison:")
    print(
        f"LS Test Acc: {ls_metrics['accuracy']:.2f}%, Test F1: {ls_metrics['macro_f1']:.2f}%"
    )
    print(
        f"SGD Test Acc: {sgd_metrics['accuracy']:.2f}%, Test F1: {sgd_metrics['macro_f1']:.2f}%"
    )

    # Prepare results dictionary
    results = {
        "hyperparameters": {
            "feature_maps": feature_maps,
            "std_dev": std_dev,
            "kernel_size": kernel_size,
            "ensemble_size": ensemble_size,
            "ls_lambda": ls_lambda,
            "sgd_lr": lr,
            "sgd_epochs": epoch,
            "seed": SEED,
        },
        "ls": {
            "accuracy": float(
                ls_metrics["accuracy"]
            ),  # Convert to float for JSON serialization
            "macro_f1": float(ls_metrics["macro_f1"]),
            "time": float(ls_time),
            "pooling": True,
            "feature_size": feature_size_ls,
        },
        "sgd": {
            "accuracy": float(sgd_metrics["accuracy"]),
            "macro_f1": float(sgd_metrics["macro_f1"]),
            "time": float(sgd_time),
            "pooling": False,
            "feature_size": feature_size_sgd,
        },
        "comparison": {
            "time_ratio": float(sgd_time / ls_time),
            "time_diff": float(ls_time - sgd_time),
            "accuracy_diff": float(ls_metrics["accuracy"] - sgd_metrics["accuracy"]),
            "f1_diff": float(ls_metrics["macro_f1"] - sgd_metrics["macro_f1"]),
        },
    }
    print(results)

    # Create save dir if not already
    os.makedirs(save_dir, exist_ok=True)

    # Save results to JSON file
    json_filename = "ls_vs_sgd_comparison.json"
    json_path = os.path.join(save_dir, json_filename)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {json_path}")

    return results


def random_search_hyperparameter_ls(
    num_steps=30,
    train_loader=None,
    test_loader=None,
    feature_maps_range=(32, 128),
    std_dev_range=(0.01, 0.5),
    kernel_sizes=[3, 5, 7],
    lambda_range=(0.001, 0.1),
    ensemble_size=(3, 10),
    seed=42,
    use_pooling=True,
    save_dir="./task2/models",
    metric_dir="./task2/metrics",
    resume_from_step=0,
    checkpoint_frequency=5,
    device="cpu",
):
    """
    Using random search and ls to find best hyperparameters, similar to task2 hyperparamters tuned: std, feature maps, kernel size, lambda (for ls)
    Using defined search space from hyperparameter arguments
    """
    # default should be cpu due to environment constraints
    print(f"Using device: {device}")

    # set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    results = []
    best_acc = 0
    best_config = None
    best_ensemble = None

    # Check if we're resuming from a previous run
    checkpoint_path = os.path.join(
        metric_dir, "elm_ls_hyperparameter_search_checkpoint.json"
    )
    if resume_from_step > 0 and os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint_data = json.load(f)
                results = checkpoint_data.get("results", [])
                best_acc = checkpoint_data.get("best_acc", 0)
                best_config = checkpoint_data.get("best_config", None)

                # Find the best result so far to set best_acc
                if results:
                    for result in results:
                        if result["accuracy"] > best_acc:
                            best_acc = result["accuracy"]
                            best_config = result.copy()

                print(
                    f"Resuming from step {resume_from_step}, loaded {len(results)} previous results"
                )
                print(f"Current best accuracy: {best_acc:.2f}%")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch")
            resume_from_step = 0

    # Continue with the random search from the specified step
    for step in range(resume_from_step, num_steps):
        # Randomly sample hyperparameters
        n_models = int(np.random.uniform(ensemble_size[0], ensemble_size[1]))
        num_feature_maps = int(
            np.exp(
                np.random.uniform(
                    np.log(feature_maps_range[0]), np.log(feature_maps_range[1])
                )
            )
        )
        std_dev = np.random.uniform(std_dev_range[0], std_dev_range[1])
        kernel_size = np.random.choice(kernel_sizes)
        lambda_reg = np.exp(
            np.random.uniform(np.log(lambda_range[0]), np.log(lambda_range[1]))
        )

        # For CIFAR-10 (32x32)
        h_out = 32 - kernel_size + 1
        w_out = 32 - kernel_size + 1
        if use_pooling:
            h_out //= 2
            w_out //= 2
        feature_size = num_feature_maps * h_out * w_out

        print(f"Step {step + 1}/{num_steps}:")
        print(
            f"Feature maps: {num_feature_maps}, Std Dev: {std_dev}, Kernel Size: {kernel_size}, Lambda: {lambda_reg}"
        )

        # Train model using least squares
        try:
            step_seed = (
                seed + step
            )  # unique seed for each step, fixed for reproducibility
            ensemble = MyEnsembleELM(
                seed=step_seed,
                n_models=n_models,
                num_feature_maps=num_feature_maps,
                std_dev=std_dev,
                kernel_size=kernel_size,
                feature_size=feature_size,
                pooling=use_pooling,
            )
            ensemble.to(device)
            start_time = time.time()
            acc_progression = []
            f1_progression = []
            for i, model in enumerate(ensemble.models):
                print(f"Training model {i + 1}/{n_models}")

                # Train with least squares
                _, _ = fit_elm_ls(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    lambda_reg=lambda_reg,
                    device=device,
                )

                # Evaluate current ensemble state
                metrics = evaluate_metrics(ensemble, test_loader, device)
                acc_progression.append(metrics["accuracy"])
                f1_progression.append(metrics["macro_f1"])

                print(
                    f"  Ensemble accuracy after {i + 1} models: {metrics['accuracy']:.2f}%"
                )

            training_time = time.time() - start_time
            final_accuracy = acc_progression[-1]
            final_f1 = f1_progression[-1]
            result = {
                "n_models": n_models,
                "num_feature_maps": num_feature_maps,
                "std_dev": std_dev,
                "lambda_reg": lambda_reg,
                "kernel_size": kernel_size,
                "pooling": use_pooling,
                "feature_size": feature_size,
                "accuracy": final_accuracy,
                "f1": final_f1,
                "training_time": training_time,
                "acc_progression": acc_progression,
                "f1_progression": f1_progression,
            }
            results.append(result)
            print(
                f"Results: Accuracy={final_accuracy:.2f}%, F1={final_f1:.2f}%, Time={training_time:.2f}s"
            )

            if final_accuracy > best_acc:
                best_acc = final_accuracy
                best_config = result.copy()
                best_ensemble = ensemble
                print(f"New best model found! Accuracy: {best_acc:.2f}%")

                # Save best ensemble models individually in a directory
                best_ensemble_dir = os.path.join(
                    save_dir, "best_hyperparameter_ls_model"
                )
                os.makedirs(best_ensemble_dir, exist_ok=True)
                for i, model in enumerate(best_ensemble.models):
                    model_file = os.path.join(best_ensemble_dir, f"model_{i}.pth")
                    torch.save(model.state_dict(), model_file)

                # Save config details for loading as JSON
                best_config_path = os.path.join(
                    save_dir, "best_hyperparameter_ls_model.json"
                )
                converted_config = convert_numpy_types(best_config)
                with open(best_config_path, "w") as f:
                    json.dump(converted_config, f, indent=4)

        except Exception as e:
            print(f"Training failed: {e}")
            continue

        # Save checkpoint periodically
        if (step + 1) % checkpoint_frequency == 0 or step == num_steps - 1:
            checkpoint_data = {
                "results": convert_numpy_types(results),
                "best_acc": float(best_acc),
                "best_config": convert_numpy_types(best_config)
                if best_config
                else None,
                "completed_steps": step + 1,
            }

            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f, indent=4)
            print(f"Checkpoint saved at step {step + 1}")

    # sort best results
    sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)

    print("\n=== Top 5 Configurations ===")
    for i, result in enumerate(sorted_results[:5]):
        print(f"Rank {i + 1}:")
        print(f"  Ensemble Size: {result['n_models']}")
        print(f"  Feature Maps: {result['num_feature_maps']}")
        print(f"  Std Dev: {result['std_dev']:.6f}")
        print(f"  Lambda: {result['lambda_reg']:.6f}")
        print(f"  Accuracy: {result['accuracy']:.2f}%")
        print(f"  F1 Score: {result['f1']:.2f}%")
        print(f"  Training Time: {result['training_time']:.2f}s")
        print()

    # Save results to json
    results_json_path = os.path.join(
        metric_dir, "elm_ls_hyperparameter_search_results.json"
    )
    converted_results = convert_numpy_types(sorted_results)
    with open(results_json_path, "w") as f:
        json.dump(converted_results, f, indent=4)

    # Save best model
    best_model_path = os.path.join(save_dir, "best_hyperparameter_model.pth")
    torch.save(best_ensemble.state_dict(), best_model_path)
    print(f"Best model saved to {best_model_path}")

    # Save config details for loading
    best_config_path = os.path.join(save_dir, "best_hyperparameter_config.json")
    converted_config = convert_numpy_types(best_config)
    with open(best_config_path, "w") as f:
        json.dump(converted_config, f, indent=4)

    # Compare with previous best
    previous_best = 57.23  # previous best ensemble accuracy
    print(f"\nPrevious best Ensemble ELM: {previous_best:.2f}%")
    print(f"New best Ensemble ELM: {best_acc:.2f}%")
    print(f"Improvement: {best_acc - previous_best:.2f}%")

    return sorted_results, best_ensemble, best_config


if __name__ == "__main__":
    print("=== Trying with best parameters from previous experiments ===")
    print(
        "Configs: feature_maps = 128, std_dev = 0.01, kernel_size=7, lr = 0.1, epoch=50"
    )

    # Comparing duration fit_lm with fit_sgd
    print("\nComparing training duration for SGD and LS...")
    comparison_duration_sgd_and_ls(
        feature_maps=128, std_dev=0.01, kernel_size=7, lr=0.1, epoch=50
    )

    print("=== Preparing to run random search for hyperparameters... ===")
    print(
        "**This will take a while to complete due to the large search space and num_steps**"
    )
    train_loader, test_loader = load_cifar10(batch_size=128)
    random_search_hyperparameter_ls(train_loader=train_loader, test_loader=test_loader)

    print("=== Producing best fit_elm_ls results for visualization... ===")
    best_fit_elm_ls_model_dir = "./task2/models/best_hyperparameter_ls_model"
    best_fit_elm_ls_model_config_file = (
        "./task2/models/best_hyperparameter_ls_model.pth"
    )

    save_path = "./task2/montage_result/new_result.png"
    visualize_model_predictions(
        best_fit_elm_ls_model_dir,
        best_fit_elm_ls_model_config_file,
        save_path=save_path,
    )
    print(f"Visualization saved to {save_path}")
    print("Task2a Done!")
